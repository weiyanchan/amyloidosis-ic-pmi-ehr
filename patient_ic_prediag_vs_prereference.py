import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

from config import IC_EXCEL_FILE, CACHE_DIR, OUTPUT_DIR, PATIENT_DATA_FILE, LAB_DATA_FILES
from matching import get_cache_key
from utils import (
    load_from_cache, parse_ic_values, parse_pipe_separated,
    filter_out_amyloid_terms, get_significance_asterisks,
)


def add_significance_bracket(ax, x1, x2, y, p_value, height_offset=0.02):
    sig_symbol = get_significance_asterisks(p_value)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_y = y + y_range * height_offset
    tick_height = y_range * 0.01

    ax.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1.5)
    ax.plot([x1, x1], [bracket_y, bracket_y - tick_height], 'k-', linewidth=1.5)
    ax.plot([x2, x2], [bracket_y, bracket_y - tick_height], 'k-', linewidth=1.5)
    ax.text((x1 + x2) / 2, bracket_y + y_range * 0.02, sig_symbol,
            ha='center', va='bottom', fontsize=12, fontweight='bold')


def load_ic_visit_data(cache_key):
    matched_control_ids_raw = load_from_cache(CACHE_DIR, cache_key, "matched_controls")
    amyloid_ids_raw = load_from_cache(CACHE_DIR, cache_key, "all_amyloid_patients")

    if matched_control_ids_raw is None or amyloid_ids_raw is None:
        raise FileNotFoundError(
            "Matched control / amyloid patient cache not found. Run matching.py first."
        )

    matched_control_ids = {p['Patient_ID'] for p in matched_control_ids_raw}

    amyl_df = pd.read_excel(IC_EXCEL_FILE, sheet_name='Amyloid_449_Visit_IC_ENTIRE')
    control_df_full = pd.read_excel(IC_EXCEL_FILE, sheet_name='Control_NEW_VARIANCE_Visit')
    control_df = control_df_full[control_df_full['Patient_ID'].isin(matched_control_ids)].copy()

    if 'Months_From_Diagnosis' in amyl_df.columns:
        amyl_df['Months_From_Reference'] = amyl_df['Months_From_Diagnosis']

    required_cols = ['Patient_ID', 'IC_Values', 'SNOMED_Descriptions', 'Months_From_Reference']
    for name, df in [('amyloidosis', amyl_df), ('control', control_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} data: {missing}")

    print(f"Amyloidosis: {len(amyl_df):,} visits, {amyl_df['Patient_ID'].nunique():,} patients")
    print(f"Matched controls: {len(control_df):,} visits, {control_df['Patient_ID'].nunique():,} patients")

    return amyl_df, control_df


def calculate_patient_ic(visit_data, patient_type, filter_amyloid_terms):
    """Patient-level mean IC, restricted to pre-diagnosis/pre-reference
    visits (Months_From_Reference < 0). If filter_amyloid_terms, IC values
    for amyloid-related SNOMED terms are excluded."""

    pre = visit_data[visit_data['Months_From_Reference'] < 0].copy()
    print(f"{patient_type}: {len(pre):,} pre-reference visits (of {len(visit_data):,} total)")

    rows = []
    n_terms_filtered, n_terms_total = 0, 0

    for patient_id, patient_visits in pre.groupby('Patient_ID'):
        patient_ic_values = []

        for _, visit in patient_visits.iterrows():
            if pd.isna(visit['IC_Values']):
                continue
            ic_values = parse_ic_values(visit['IC_Values'])

            if filter_amyloid_terms and pd.notna(visit['SNOMED_Descriptions']):
                descriptions = parse_pipe_separated(visit['SNOMED_Descriptions'])
                n_terms_total += len(ic_values)
                ic_values, n_filtered = filter_out_amyloid_terms(ic_values, descriptions)
                n_terms_filtered += n_filtered

            patient_ic_values.extend(ic_values)

        if patient_ic_values:
            rows.append({
                'Patient_ID': str(patient_id),
                'Patient_Type': patient_type,
                'Total_Visits': len(patient_visits),
                'Total_SNOMED_Terms': len(patient_ic_values),
                'Mean_IC_Per_Patient': np.mean(patient_ic_values),
            })

    df = pd.DataFrame(rows)
    print(f"{patient_type}: {len(df):,} patients with valid IC data"
          + (f" ({n_terms_filtered:,}/{n_terms_total:,} amyloid terms filtered)"
             if filter_amyloid_terms and n_terms_total else ""))
    return df


def plot_distribution(amyl_ic, control_ic, filename):
    amyl, control = pd.Series(amyl_ic).dropna(), pd.Series(control_ic).dropna()
    if amyl.empty or control.empty:
        print("Insufficient data for distribution plot")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    bins = min(50, max(20, int(np.sqrt(len(amyl)))))

    ax.hist(control, bins=bins, density=True, alpha=0.4, color='#4472C4',
            label=f'Control (n={len(control):,})', edgecolor='black', linewidth=0.5)
    ax.hist(amyl, bins=bins, density=True, alpha=0.4, color='#E74C3C',
            label=f'Amyloidosis (n={len(amyl):,})', edgecolor='black', linewidth=0.5)

    x_range = np.linspace(0, max(control.max(), amyl.max()) * 1.05, 400)
    for data, color, name in [(control, '#4472C4', 'Control'), (amyl, '#E74C3C', 'Amyloidosis')]:
        if len(data) > 1:
            ax.plot(x_range, gaussian_kde(data)(x_range), color=color, linewidth=3, alpha=0.8,
                    label=f'{name} Density')
            ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'{name} Mean: {data.mean():.4f}')

    ax.set_title('Distribution of Patient-Level SNOMED Terms IC\n'
                 'Amyloidosis Pre-Diagnosis vs Control Pre-Reference',
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Patient-Level Average SNOMED Terms IC', fontsize=15)
    ax.set_ylabel('Density', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_boxplot(amyl_ic, control_ic, p_value, filename):
    amyl, control = pd.Series(amyl_ic).dropna(), pd.Series(control_ic).dropna()
    if amyl.empty or control.empty:
        print("Insufficient data for box plot")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    labels = [f'Control\n(Pre-Reference)\n(n={len(control):,})',
              f'Amyloidosis\n(Pre-Diagnosis)\n(n={len(amyl):,})']

    bp = ax.boxplot([control, amyl], labels=labels, patch_artist=True,
                     boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5), medianprops=dict(linewidth=2, color='black'))

    for patch, color in zip(bp['boxes'], ['#4472C4', '#E74C3C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    ax.scatter([1], [control.mean()], marker='x', s=120, c='black', linewidth=1.5, zorder=10)
    ax.scatter([2], [amyl.mean()], marker='x', s=120, c='black', linewidth=1.5, zorder=10)

    if pd.notna(p_value):
        current_ylim = ax.get_ylim()
        add_significance_bracket(ax, 1, 2, current_ylim[1], p_value)
        new_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], new_ylim[1] + (new_ylim[1] - new_ylim[0]) * 0.05)

    ax.set_title('Patient-Level SNOMED Terms IC Comparison\n'
                 'Amyloidosis Pre-Diagnosis vs Control Pre-Reference',
                 fontsize=18, fontweight='bold', pad=35)
    ax.set_ylabel('Patient-Level Average SNOMED Terms IC', fontsize=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def main():
    cache_key = get_cache_key(PATIENT_DATA_FILE, LAB_DATA_FILES)
    amyl_visits, control_visits = load_ic_visit_data(cache_key)

    amyl_ic = calculate_patient_ic(amyl_visits, 'Amyloidosis', filter_amyloid_terms=False)
    control_ic = calculate_patient_ic(control_visits, 'Control', filter_amyloid_terms=True)

    if amyl_ic.empty or control_ic.empty:
        print("No data available for comparison")
        return None

    amyl_vals = amyl_ic['Mean_IC_Per_Patient']
    ctrl_vals = control_ic['Mean_IC_Per_Patient']
    _, p_value = mannwhitneyu(ctrl_vals, amyl_vals, alternative='two-sided')

    print(f"Amyloidosis pre-diagnosis (n={len(amyl_vals):,}): {amyl_vals.mean():.6f} +/- {amyl_vals.std():.6f}")
    print(f"Control pre-reference (n={len(ctrl_vals):,}): {ctrl_vals.mean():.6f} +/- {ctrl_vals.std():.6f}")
    print(f"P-value: {p_value:.2e} ({get_significance_asterisks(p_value)})")

    output_file = os.path.join(OUTPUT_DIR, "patient_ic_prediag_vs_prereference.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        amyl_ic.to_excel(writer, sheet_name='Amyloidosis_PreDiag_IC', index=False)
        control_ic.to_excel(writer, sheet_name='Control_PreRef_IC', index=False)
        pd.concat([amyl_ic, control_ic], ignore_index=True).to_excel(writer, sheet_name='Combined', index=False)

        summary = pd.DataFrame([
            ['Comparison', 'Amyloidosis Pre-Diagnosis vs Control Pre-Reference'],
            ['Amyloidosis patients', len(amyl_ic)],
            ['Control patients', len(control_ic)],
            ['Amyloidosis mean IC', amyl_vals.mean()],
            ['Control mean IC', ctrl_vals.mean()],
            ['Mann-Whitney p-value', p_value],
        ])
        summary.to_excel(writer, sheet_name='Summary', index=False, header=False)

    print(f"Results saved to {output_file}")

    plot_distribution(amyl_vals, ctrl_vals,
                       os.path.join(OUTPUT_DIR, "patient_ic_prediag_vs_prereference_distribution.png"))
    plot_boxplot(amyl_vals, ctrl_vals, p_value,
                 os.path.join(OUTPUT_DIR, "patient_ic_prediag_vs_prereference_boxplot.png"))

    return {'amyl': amyl_ic, 'control': control_ic, 'p_value': p_value}


if __name__ == "__main__":
    main()
