"""
Merges SNOMED-coded and lab-derived renal findings (proteinuria,
nephrotic-range proteinuria, microscopic haematuria) into a single
per-visit table for AL amyloidosis patients, combining findings by source
and computing patient- and finding-level summaries. Output feeds
figure5_renal_chest_pain_timeline.py.
"""

import os
import pandas as pd
import numpy as np

from config import RENAL_TERMS_FILENAME, LAB_RENAL_FINDINGS_FILENAME, COMBINED_RENAL_FILENAME, OUTPUT_DIR
from utils import parse_pipe_separated

RENAL_TERMS_FILE = os.path.join(OUTPUT_DIR, RENAL_TERMS_FILENAME)
LAB_RENAL_FINDINGS_FILE = os.path.join(OUTPUT_DIR, LAB_RENAL_FINDINGS_FILENAME)

MERGE_COLUMNS = ['Patient_ID', 'Visit_ID', 'Visit_Date', 'First_Amyloid_Date',
                  'Months_From_Diagnosis', 'Matching_Terms']


def load_data():
    snomed_df = pd.read_excel(RENAL_TERMS_FILE, sheet_name='Renal_Terms_Visits')
    print(f"SNOMED renal findings: {len(snomed_df):,} visits, {snomed_df['Patient_ID'].nunique()} patients")

    lab_df = pd.read_excel(LAB_RENAL_FINDINGS_FILE, sheet_name='Lab_Renal_Findings_Visits')
    print(f"Lab-derived renal findings: {len(lab_df):,} visits, {lab_df['Patient_ID'].nunique()} patients")

    return snomed_df, lab_df


def merge_findings(snomed_df, lab_df):
    snomed_df = snomed_df.rename(columns={'Matching_SNOMED_Terms': 'Matching_Terms'})
    lab_df = lab_df.rename(columns={'Matching_SNOMED_Terms': 'Matching_Terms'})

    merged_df = pd.merge(
        snomed_df[MERGE_COLUMNS],
        lab_df[MERGE_COLUMNS + ['TPU', 'PU', 'P24U', 'URBC']],
        on='Visit_ID', how='outer', suffixes=('_SNOMED', '_Lab'),
    )

    # Coalesce shared identifying columns (present under both suffixes)
    for col in ['Patient_ID', 'Visit_Date', 'First_Amyloid_Date', 'Months_From_Diagnosis']:
        merged_df[col] = merged_df[f'{col}_SNOMED'].fillna(merged_df[f'{col}_Lab'])
        merged_df = merged_df.drop(columns=[f'{col}_SNOMED', f'{col}_Lab'])

    def combine_terms(row):
        terms = [str(row[c]) for c in ('Matching_Terms_SNOMED', 'Matching_Terms_Lab')
                 if pd.notna(row[c]) and row[c] != '']
        return ' | '.join(terms)

    def classify_source(row):
        has_snomed = pd.notna(row['Matching_Terms_SNOMED']) and row['Matching_Terms_SNOMED'] != ''
        has_lab = pd.notna(row['Matching_Terms_Lab']) and row['Matching_Terms_Lab'] != ''
        if has_snomed and has_lab:
            return 'SNOMED + Lab'
        if has_snomed:
            return 'SNOMED only'
        if has_lab:
            return 'Lab only'
        return 'Unknown'

    merged_df['All_Renal_Findings'] = merged_df.apply(combine_terms, axis=1)
    merged_df['Data_Source'] = merged_df.apply(classify_source, axis=1)

    final_columns = ['Patient_ID', 'Visit_ID', 'Visit_Date', 'First_Amyloid_Date',
                      'Months_From_Diagnosis', 'All_Renal_Findings', 'Data_Source',
                      'Matching_Terms_SNOMED', 'Matching_Terms_Lab', 'TPU', 'PU', 'P24U', 'URBC']
    merged_df = merged_df[final_columns].sort_values(['Patient_ID', 'Months_From_Diagnosis'])

    print(f"Merged: {len(merged_df):,} visits "
          f"(SNOMED only: {(merged_df['Data_Source'] == 'SNOMED only').sum():,}, "
          f"Lab only: {(merged_df['Data_Source'] == 'Lab only').sum():,}, "
          f"both: {(merged_df['Data_Source'] == 'SNOMED + Lab').sum():,})")

    return merged_df


def create_patient_summary(merged_df):
    rows = []

    for patient_id, patient_data in merged_df.groupby('Patient_ID'):
        all_findings, snomed_findings, lab_findings = set(), set(), set()
        for _, row in patient_data.iterrows():
            all_findings.update(parse_pipe_separated(row['All_Renal_Findings']))
            snomed_findings.update(parse_pipe_separated(row['Matching_Terms_SNOMED']))
            lab_findings.update(parse_pipe_separated(row['Matching_Terms_Lab']))

        valid_months = patient_data['Months_From_Diagnosis'].dropna()
        if len(valid_months):
            earliest = patient_data.loc[valid_months.idxmin()]
            earliest_months, earliest_findings, earliest_source = (
                earliest['Months_From_Diagnosis'], earliest['All_Renal_Findings'], earliest['Data_Source']
            )
        else:
            earliest_months, earliest_findings, earliest_source = np.nan, '', ''

        rows.append({
            'Patient_ID': patient_id,
            'Total_Visits': len(patient_data),
            'SNOMED_Only_Visits': (patient_data['Data_Source'] == 'SNOMED only').sum(),
            'Lab_Only_Visits': (patient_data['Data_Source'] == 'Lab only').sum(),
            'Both_Sources_Visits': (patient_data['Data_Source'] == 'SNOMED + Lab').sum(),
            'All_Unique_Findings': ' | '.join(sorted(all_findings)),
            'SNOMED_Findings': ' | '.join(sorted(snomed_findings)),
            'Lab_Findings': ' | '.join(sorted(lab_findings)),
            'Earliest_Occurrence_Months': earliest_months,
            'Earliest_Findings': earliest_findings,
            'Earliest_Source': earliest_source,
            'First_Amyloid_Date': patient_data['First_Amyloid_Date'].iloc[0],
        })

    summary_df = pd.DataFrame(rows).sort_values('Earliest_Occurrence_Months', na_position='last')
    print(f"Patient summary: {len(summary_df):,} patients")
    return summary_df


def create_comparison_statistics(merged_df, patient_summary_df):
    stats = {
        'Total_Patients': len(patient_summary_df),
        'Total_Visits': len(merged_df),
        'SNOMED_Only_Visits': (merged_df['Data_Source'] == 'SNOMED only').sum(),
        'Lab_Only_Visits': (merged_df['Data_Source'] == 'Lab only').sum(),
        'Both_Sources_Visits': (merged_df['Data_Source'] == 'SNOMED + Lab').sum(),
        'Patients_SNOMED_Only': (patient_summary_df['Lab_Only_Visits'] == 0).sum(),
        'Patients_Lab_Only': (patient_summary_df['SNOMED_Only_Visits'] == 0).sum(),
        'Patients_Both_Sources': ((patient_summary_df['SNOMED_Only_Visits'] > 0)
                                   & (patient_summary_df['Lab_Only_Visits'] > 0)).sum(),
    }
    print(f"Patients: {stats['Total_Patients']}, Visits: {stats['Total_Visits']} "
          f"(SNOMED-only patients: {stats['Patients_SNOMED_Only']}, "
          f"Lab-only patients: {stats['Patients_Lab_Only']}, "
          f"both: {stats['Patients_Both_Sources']})")
    return pd.DataFrame([stats])


def create_finding_frequency(merged_df):
    all_findings = [
        f for findings in merged_df['All_Renal_Findings']
        for f in parse_pipe_separated(findings)
    ]

    frequency_df = pd.Series(all_findings).value_counts().reset_index()
    frequency_df.columns = ['Finding', 'Frequency']
    frequency_df['Source'] = frequency_df['Finding'].apply(
        lambda f: 'Lab-derived' if '(Lab)' in f else 'SNOMED'
    )

    print(f"Unique findings: {len(frequency_df)}")
    return frequency_df


def export_results(merged_df, patient_summary_df, stats_df, frequency_df, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name='Combined_All_Visits', index=False)
        patient_summary_df.to_excel(writer, sheet_name='Patient_Summary', index=False)
        stats_df.to_excel(writer, sheet_name='Comparison_Statistics', index=False)
        frequency_df.to_excel(writer, sheet_name='Finding_Frequency', index=False)

        for label, source in [('SNOMED_Only_Visits', 'SNOMED only'),
                               ('Lab_Only_Visits', 'Lab only'),
                               ('Both_Sources_Visits', 'SNOMED + Lab')]:
            subset = merged_df[merged_df['Data_Source'] == source]
            if len(subset):
                subset.to_excel(writer, sheet_name=label, index=False)

        bins = [-float('inf'), -12, -6, 0, 6, 12, float('inf')]
        labels = ['< -12 months', '-12 to -6 months', '-6 to 0 months',
                  '0 to 6 months', '6 to 12 months', '> 12 months']
        merged_df = merged_df.copy()
        merged_df['Temporal_Bin'] = pd.cut(merged_df['Months_From_Diagnosis'], bins=bins, labels=labels)
        merged_df.groupby(['Temporal_Bin', 'Data_Source']).size().reset_index(name='Count').to_excel(
            writer, sheet_name='Temporal_Distribution', index=False
        )

    print(f"Exported: {output_file}")


def main():
    snomed_df, lab_df = load_data()
    merged_df = merge_findings(snomed_df, lab_df)
    patient_summary_df = create_patient_summary(merged_df)
    stats_df = create_comparison_statistics(merged_df, patient_summary_df)
    frequency_df = create_finding_frequency(merged_df)

    output_file = os.path.join(OUTPUT_DIR, COMBINED_RENAL_FILENAME)
    export_results(merged_df, patient_summary_df, stats_df, frequency_df, output_file)


if __name__ == "__main__":
    main()
