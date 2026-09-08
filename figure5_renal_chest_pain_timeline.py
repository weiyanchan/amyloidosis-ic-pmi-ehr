"""
Figure 5: timeline of pre-diagnosis renal (proteinuria, nephrotic-range
proteinuria, microscopic haematuria) and chest pain findings in AL
amyloidosis patients, relative to diagnosis date.

Requires COMBINED_RENAL_FILE (config.py) -- a precomputed workbook
combining SNOMED-coded and lab-derived renal findings per visit (sheet
'Combined_All_Visits', columns including Patient_ID, Months_From_Diagnosis,
All_Renal_Findings, Matching_Terms_SNOMED, Matching_Terms_Lab). This file
is produced by a separate extraction step not yet included in this
repository.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

from config import COMBINED_RENAL_FILENAME, IC_EXCEL_FILE, CLASSIFICATION_FILE, OUTPUT_DIR, AL_SUBTYPES

COMBINED_RENAL_FILE = os.path.join(OUTPUT_DIR, COMBINED_RENAL_FILENAME)

CHEST_PAIN_TERMS = ['chest pain', 'atypical chest pain']

# Horizontal offset (months) so overlapping finding types are visible side by side
X_DODGE = {'Proteinuria': -0.15, 'Nephrotic': -0.05, 'Hematuria': 0.05, 'Chest Pain': 0.15}

MARKER_STYLE = {
    'Proteinuria': dict(color='blue', marker='o'),
    'Nephrotic': dict(color='orange', marker='s'),
    'Hematuria': dict(color='red', marker='^'),
}


def load_al_patient_ids():
    df = pd.read_excel(CLASSIFICATION_FILE, sheet_name='NEW_Amyloid_Patients')
    patient_ids = df.loc[df['Classified_Subtype'].isin(AL_SUBTYPES), 'Patient_ID'].astype(str).tolist()
    print(f"AL amyloidosis patients: {len(patient_ids)}")
    return patient_ids


def load_combined_renal_data():
    visits_df = pd.read_excel(COMBINED_RENAL_FILE, sheet_name='Combined_All_Visits')
    print(f"Loaded {len(visits_df):,} visits with renal findings "
          f"from {visits_df['Patient_ID'].nunique()} patients")
    return visits_df


def load_chest_pain_data(al_patient_ids):
    visits_df = pd.read_excel(IC_EXCEL_FILE, sheet_name='Amyloid_449_Visit_IC_ENTIRE')
    visits_df['Patient_ID'] = visits_df['Patient_ID'].astype(str)
    al_visits = visits_df[visits_df['Patient_ID'].isin(al_patient_ids)]

    chest_pain_visits = []
    for _, row in al_visits.iterrows():
        descriptions = row['SNOMED_Descriptions']
        if pd.isna(descriptions) or descriptions == '':
            continue
        if any(term in str(descriptions).lower() for term in CHEST_PAIN_TERMS):
            chest_pain_visits.append({
                'Patient_ID': row['Patient_ID'],
                'Visit_ID': row['Visit_ID'],
                'Visit_Date': row['Visit_Date'],
                'First_Amyloid_Date': row['First_Amyloid_Date'],
                'Months_From_Diagnosis': row['Months_From_Diagnosis'],
                'Finding_Type': 'Chest Pain',
                'Source': 'SNOMED',
            })

    chest_pain_df = pd.DataFrame(chest_pain_visits)
    n_patients = chest_pain_df['Patient_ID'].nunique() if len(chest_pain_df) else 0
    print(f"Chest pain visits: {len(chest_pain_df):,} from {n_patients} patients")
    return chest_pain_df


def classify_finding_type(row):
    """Classifies a visit's renal findings into Proteinuria/Nephrotic/
    Hematuria, each tagged with its source (SNOMED, Lab, or Both).
    Nephrotic-range proteinuria takes precedence over plain proteinuria."""

    all_findings = row['All_Renal_Findings']
    if pd.isna(all_findings) or all_findings == '':
        return []

    snomed_findings = str(row['Matching_Terms_SNOMED']).lower() if pd.notna(row['Matching_Terms_SNOMED']) else ''
    lab_findings = str(row['Matching_Terms_Lab']).lower() if pd.notna(row['Matching_Terms_Lab']) else ''
    all_lower = str(all_findings).lower()

    def source_for(keyword):
        in_snomed = keyword in snomed_findings
        in_lab = keyword in lab_findings
        if in_snomed and in_lab:
            return 'Both'
        if in_snomed:
            return 'SNOMED'
        if in_lab:
            return 'Lab'
        return None

    findings = []
    if 'nephrotic' in all_lower:
        source = source_for('nephrotic')
        if source:
            findings.append(('Nephrotic', source))
    elif 'proteinuria' in all_lower:
        source = source_for('proteinuria')
        if source:
            findings.append(('Proteinuria', source))

    if 'hematuria' in all_lower or 'haematuria' in all_lower:
        in_snomed = 'hematuria' in snomed_findings or 'haematuria' in snomed_findings
        in_lab = 'hematuria' in lab_findings or 'haematuria' in lab_findings
        source = 'Both' if in_snomed and in_lab else 'SNOMED' if in_snomed else 'Lab' if in_lab else None
        if source:
            findings.append(('Hematuria', source))

    return findings


def combine_renal_and_chest_pain(renal_df, chest_pain_df):
    renal_pre = renal_df[renal_df['Months_From_Diagnosis'] < 0].copy()
    renal_pre['Finding_Classification'] = renal_pre.apply(classify_finding_type, axis=1)
    renal_pre = renal_pre[renal_pre['Finding_Classification'].apply(len) > 0]

    chest_pain_pre = (
        chest_pain_df[chest_pain_df['Months_From_Diagnosis'] < 0].copy()
        if len(chest_pain_df) else pd.DataFrame()
    )

    print(f"Pre-diagnosis: {len(renal_pre)} renal visits, {len(chest_pain_pre)} chest pain visits")
    return renal_pre, chest_pain_pre


def _patient_earliest_months(patient_id, renal_df, chest_pain_df):
    months = []
    if len(renal_df):
        months.extend(renal_df.loc[renal_df['Patient_ID'] == patient_id, 'Months_From_Diagnosis'])
    if len(chest_pain_df):
        months.extend(chest_pain_df.loc[chest_pain_df['Patient_ID'] == patient_id, 'Months_From_Diagnosis'])
    return months


def create_combined_timeline(renal_df, chest_pain_df, show_patient_ids, output_file):
    renal_patients = set(renal_df['Patient_ID'].unique()) if len(renal_df) else set()
    chest_patients = set(chest_pain_df['Patient_ID'].unique()) if len(chest_pain_df) else set()
    all_patients = renal_patients | chest_patients

    if not all_patients:
        print("No patients with findings to plot")
        return

    patient_earliest = {
        pid: min(_patient_earliest_months(pid, renal_df, chest_pain_df))
        for pid in all_patients if _patient_earliest_months(pid, renal_df, chest_pain_df)
    }
    sorted_patients = sorted(patient_earliest.items(), key=lambda x: x[1])
    patient_to_row = {pid: i for i, (pid, _) in enumerate(sorted_patients)}

    fig, ax = plt.subplots(figsize=(18, max(10, len(patient_to_row) * 0.4)))

    for patient_id, row_idx in patient_to_row.items():
        patient_renal = renal_df[renal_df['Patient_ID'] == patient_id] if len(renal_df) else pd.DataFrame()
        patient_chest = chest_pain_df[chest_pain_df['Patient_ID'] == patient_id] if len(chest_pain_df) else pd.DataFrame()

        all_months = list(patient_renal.get('Months_From_Diagnosis', [])) + \
                     list(patient_chest.get('Months_From_Diagnosis', []))
        if all_months:
            ax.plot([min(all_months), 0], [row_idx, row_idx], color='gray', alpha=0.3, linewidth=2, zorder=1)

        for _, visit in patient_renal.iterrows():
            base_month = visit['Months_From_Diagnosis']
            for finding_type, source in visit['Finding_Classification']:
                style = MARKER_STYLE.get(finding_type)
                if style is None:
                    continue

                x_plot = base_month + X_DODGE[finding_type]
                if source in ('SNOMED', 'Both'):
                    ax.scatter(x_plot, row_idx, color=style['color'], marker=style['marker'],
                               s=120, alpha=0.8, edgecolors='black', linewidths=1.5, zorder=3)
                else:
                    ax.scatter(x_plot, row_idx, facecolors='none', edgecolors=style['color'],
                               marker=style['marker'], s=120, alpha=0.8, linewidths=2, zorder=3)

        for _, visit in patient_chest.iterrows():
            x_plot = visit['Months_From_Diagnosis'] + X_DODGE['Chest Pain']
            ax.scatter(x_plot, row_idx, color='green', marker='X', s=90, alpha=0.85,
                       edgecolors='black', linewidths=1.2, zorder=4)

    ax.set_xlabel('Months Before Diagnosis', fontweight='bold', fontsize=12)
    ax.set_ylabel('Patient', fontweight='bold', fontsize=12)
    ax.set_title(
        'AL Amyloidosis - Pre-Diagnosis Renal & Cardiac Findings Timeline\n'
        'Blue circles = Proteinuria | Orange squares = Nephrotic range | Red triangles = Haematuria | Green X = Chest pain\n'
        'Filled = SNOMED coded | Unfilled = Lab-derived only',
        fontweight='bold', fontsize=13, pad=15,
    )
    ax.axvline(0, color='black', linestyle='--', linewidth=2.5, alpha=0.8, label='Diagnosis Date', zorder=3)
    ax.grid(True, alpha=0.3, axis='x')

    ax.set_yticks(range(len(patient_to_row)))
    if show_patient_ids:
        ax.set_yticklabels([pid for pid, _ in sorted_patients], fontsize=8)
    else:
        ax.set_yticklabels([f'Patient {i + 1}' for i in range(len(patient_to_row))], fontsize=8)

    xlim = ax.get_xlim()
    ax.set_xlim(min(xlim[0], -1), 1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, label='Proteinuria (SNOMED)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=10,
               markeredgecolor='blue', markeredgewidth=2, label='Proteinuria (Lab only)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, label='Nephrotic range proteinuria (SNOMED)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markersize=10,
               markeredgecolor='orange', markeredgewidth=2, label='Nephrotic range proteinuria (Lab only)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, label='Microscopic haematuria (SNOMED)', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markersize=10,
               markeredgecolor='red', markeredgewidth=2, label='Microscopic haematuria (Lab only)', linestyle='None'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markersize=9,
               markeredgecolor='black', markeredgewidth=1.2, label='Chest pain (SNOMED)', linestyle='None'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2.5, label='Diagnosis Date'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.3, label='Duration to diagnosis'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95, bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")
    print(f"Patients plotted: {len(patient_to_row)} "
          f"(renal: {len(renal_patients)}, chest pain: {len(chest_patients)}, both: {len(renal_patients & chest_patients)})")


def create_summary_statistics(renal_df, chest_pain_df):
    all_patients = set()
    if len(renal_df):
        all_patients.update(renal_df['Patient_ID'].unique())
    if len(chest_pain_df):
        all_patients.update(chest_pain_df['Patient_ID'].unique())

    rows = []
    for patient_id in all_patients:
        patient_renal = renal_df[renal_df['Patient_ID'] == patient_id] if len(renal_df) else pd.DataFrame()
        patient_chest = chest_pain_df[chest_pain_df['Patient_ID'] == patient_id] if len(chest_pain_df) else pd.DataFrame()

        all_findings = [f for findings in patient_renal.get('Finding_Classification', []) for f in findings]
        months = _patient_earliest_months(patient_id, renal_df, chest_pain_df)

        rows.append({
            'Patient_ID': patient_id,
            'Has_Proteinuria': any(f[0] == 'Proteinuria' for f in all_findings),
            'Has_Nephrotic': any(f[0] == 'Nephrotic' for f in all_findings),
            'Has_Hematuria': any(f[0] == 'Hematuria' for f in all_findings),
            'Has_Chest_Pain': len(patient_chest) > 0,
            'Earliest_Month': min(months) if months else np.nan,
            'Total_Visits': len(patient_renal) + len(patient_chest),
        })

    summary_df = pd.DataFrame(rows).sort_values('Earliest_Month', na_position='last')

    print(f"Patients: {len(summary_df)} "
          f"(proteinuria={summary_df['Has_Proteinuria'].sum()}, "
          f"nephrotic={summary_df['Has_Nephrotic'].sum()}, "
          f"hematuria={summary_df['Has_Hematuria'].sum()}, "
          f"chest_pain={summary_df['Has_Chest_Pain'].sum()})")
    print(f"Earliest finding: mean={summary_df['Earliest_Month'].mean():.1f}, "
          f"median={summary_df['Earliest_Month'].median():.1f} months before diagnosis "
          f"(range {summary_df['Earliest_Month'].min():.1f} to {summary_df['Earliest_Month'].max():.1f})")

    return summary_df


def main():
    al_patient_ids = load_al_patient_ids()
    renal_df = load_combined_renal_data()
    chest_pain_df = load_chest_pain_data(al_patient_ids)
    renal_pre, chest_pain_pre = combine_renal_and_chest_pain(renal_df, chest_pain_df)

    create_combined_timeline(renal_pre, chest_pain_pre, show_patient_ids=False,
                              output_file=os.path.join(OUTPUT_DIR, 'combined_findings_timeline.png'))
    create_combined_timeline(renal_pre, chest_pain_pre, show_patient_ids=True,
                              output_file=os.path.join(OUTPUT_DIR, 'combined_findings_timeline_with_ids.png'))

    summary_df = create_summary_statistics(renal_pre, chest_pain_pre)

    output_file = os.path.join(OUTPUT_DIR, 'combined_findings_summary.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Patient_Summary', index=False)
    print(f"Summary exported: {output_file}")


if __name__ == "__main__":
    main()
