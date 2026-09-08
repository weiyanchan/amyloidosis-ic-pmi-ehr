"""
Extracts AL amyloidosis visits with lab-derived renal findings
(proteinuria, nephrotic-range proteinuria, microscopic haematuria), using
the thresholds defined once in config.py. Output feeds
merge_renal_findings.py.
"""

import os
import pandas as pd
import numpy as np

from config import (
    CLASSIFICATION_FILE, LAB_FILE, OUTPUT_DIR, LAB_RENAL_FINDINGS_FILENAME,
    PROTEINURIA_THRESHOLD, NEPHROTIC_THRESHOLD, MICROSCOPIC_HAEMATURIA_THRESHOLD, AL_SUBTYPES,
)
from utils import parse_pipe_separated


def load_al_amyloidosis_patients():
    df = pd.read_excel(CLASSIFICATION_FILE, sheet_name='NEW_Amyloid_Patients')
    al_patients = df[df['Classified_Subtype'].isin(AL_SUBTYPES)]
    patient_ids = al_patients['Patient_ID'].astype(str).tolist()

    for subtype in AL_SUBTYPES:
        print(f"  {subtype}: {(al_patients['Classified_Subtype'] == subtype).sum()} patients")
    print(f"Total AL amyloidosis patients: {len(patient_ids)}")

    return patient_ids


def parse_lab_value(value_str):
    """Parses a lab value string to a float. Takes the first value if
    multiple are pipe-separated, and strips comparison operators/commas."""
    if pd.isna(value_str) or value_str == '':
        return np.nan

    value_str = str(value_str)
    if '|' in value_str:
        value_str = value_str.split('|')[0].strip()

    try:
        return float(value_str.replace(',', '').replace('>', '').replace('<', '').strip())
    except ValueError:
        return np.nan


def classify_visit_findings(tpu, pu, p24u, urbc):
    """Classifies a visit's lab values into renal finding terms, per the
    thresholds defined in config.py. Nephrotic-range proteinuria takes
    precedence over plain proteinuria for the same visit."""

    terms = []
    proteinuria_value = tpu if not np.isnan(tpu) else pu

    nephrotic = (not np.isnan(proteinuria_value) and proteinuria_value > NEPHROTIC_THRESHOLD) or \
                (not np.isnan(p24u) and p24u > NEPHROTIC_THRESHOLD)

    if nephrotic:
        terms.append('Nephrotic range proteinuria (Lab)')
    else:
        proteinuria = (not np.isnan(proteinuria_value) and proteinuria_value > PROTEINURIA_THRESHOLD) or \
                      (not np.isnan(p24u) and p24u > PROTEINURIA_THRESHOLD)
        if proteinuria:
            terms.append('Proteinuria (Lab)')

    if not np.isnan(urbc) and urbc >= MICROSCOPIC_HAEMATURIA_THRESHOLD:
        terms.append('Microscopic haematuria (Lab)')

    return terms


def extract_lab_renal_findings(patient_ids):
    al_labs = pd.read_excel(LAB_FILE, sheet_name='AL_Amyloidosis')
    al_labs['Patient_ID'] = al_labs['Patient_ID'].astype(str)
    al_visits = al_labs[al_labs['Patient_ID'].isin(patient_ids)]
    print(f"AL amyloidosis lab visits: {len(al_labs):,} total, {len(al_visits):,} from classified patients")

    extracted = []
    for _, row in al_visits.iterrows():
        tpu = parse_lab_value(row.get('TPU', np.nan))
        pu = parse_lab_value(row.get('PU', np.nan))
        p24u = parse_lab_value(row.get('P24U', np.nan))
        urbc = parse_lab_value(row.get('URBC', np.nan))

        terms = classify_visit_findings(tpu, pu, p24u, urbc)
        if not terms:
            continue

        extracted.append({
            'Patient_ID': row['Patient_ID'],
            'Visit_ID': row['Visit_ID'],
            'Visit_Date': row.get('Visit_Date', ''),
            'First_Amyloid_Date': row.get('Reference_Date', ''),
            'Months_From_Diagnosis': row.get('Months_From_Reference', np.nan),
            'Matching_SNOMED_Terms': ' | '.join(terms),
            'TPU': tpu if not np.isnan(tpu) else '',
            'PU': pu if not np.isnan(pu) else '',
            'P24U': p24u if not np.isnan(p24u) else '',
            'URBC': urbc if not np.isnan(urbc) else '',
        })

    result_df = pd.DataFrame(extracted)
    if result_df.empty:
        print("No visits with lab-derived renal findings found")
        return result_df

    result_df = result_df.sort_values(['Patient_ID', 'Months_From_Diagnosis'])

    n_proteinuria = result_df['Matching_SNOMED_Terms'].str.contains(r'Proteinuria \(Lab\)', regex=True).sum()
    n_nephrotic = result_df['Matching_SNOMED_Terms'].str.contains(r'Nephrotic range proteinuria \(Lab\)', regex=True).sum()
    n_haematuria = result_df['Matching_SNOMED_Terms'].str.contains(r'Microscopic haematuria \(Lab\)', regex=True).sum()

    print(f"Visits with lab-derived renal findings: {len(result_df)} "
          f"({result_df['Patient_ID'].nunique()} patients)")
    print(f"  Proteinuria: {n_proteinuria}, Nephrotic: {n_nephrotic}, Haematuria: {n_haematuria}")

    n_pre = (result_df['Months_From_Diagnosis'] < 0).sum()
    n_at = (result_df['Months_From_Diagnosis'] == 0).sum()
    n_post = (result_df['Months_From_Diagnosis'] > 0).sum()
    print(f"  Pre-diagnosis: {n_pre}, at diagnosis: {n_at}, post-diagnosis: {n_post}")

    return result_df


def export_results(result_df, output_file):
    if result_df.empty:
        print("No data to export")
        return

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Lab_Renal_Findings_Visits', index=False)

        patient_summary = result_df.groupby('Patient_ID').agg(
            Total_Visits_With_Lab_Findings=('Visit_ID', 'count'),
            First_Occurrence_Months=('Months_From_Diagnosis', 'min'),
            Last_Occurrence_Months=('Months_From_Diagnosis', 'max'),
            All_Unique_Lab_Findings=('Matching_SNOMED_Terms', lambda x: ' || '.join(x.unique())),
        ).reset_index()
        patient_summary.to_excel(writer, sheet_name='Patient_Summary', index=False)

        all_terms = [t for terms in result_df['Matching_SNOMED_Terms'] for t in parse_pipe_separated(terms)]
        term_counts = pd.Series(all_terms).value_counts().reset_index()
        term_counts.columns = ['Lab_Finding', 'Frequency']
        term_counts.to_excel(writer, sheet_name='Finding_Frequency', index=False)

        bins = [-float('inf'), -12, -6, 0, 6, 12, float('inf')]
        labels = ['< -12 months', '-12 to -6 months', '-6 to 0 months',
                  '0 to 6 months', '6 to 12 months', '> 12 months']
        result_df = result_df.copy()
        result_df['Temporal_Bin'] = pd.cut(result_df['Months_From_Diagnosis'], bins=bins, labels=labels)
        temporal_dist = result_df['Temporal_Bin'].value_counts().sort_index().reset_index()
        temporal_dist.columns = ['Time_Period', 'Visit_Count']
        temporal_dist.to_excel(writer, sheet_name='Temporal_Distribution', index=False)

    print(f"Exported: {output_file}")


def main():
    print(f"Thresholds: proteinuria >{PROTEINURIA_THRESHOLD} g/day, "
          f"nephrotic-range >{NEPHROTIC_THRESHOLD} g/day, "
          f"microscopic haematuria >={MICROSCOPIC_HAEMATURIA_THRESHOLD} RBC/HPF")

    patient_ids = load_al_amyloidosis_patients()
    result_df = extract_lab_renal_findings(patient_ids)

    output_file = os.path.join(OUTPUT_DIR, LAB_RENAL_FINDINGS_FILENAME)
    export_results(result_df, output_file)


if __name__ == "__main__":
    main()
