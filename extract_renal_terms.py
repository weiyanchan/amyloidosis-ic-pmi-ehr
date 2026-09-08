"""
Extracts AL amyloidosis visits with SNOMED-coded renal terms (haematuria/
hematuria/proteinuria, in any documented variant). Output feeds
merge_renal_findings.py.
"""

import os
import pandas as pd

from config import CLASSIFICATION_FILE, IC_EXCEL_FILE, OUTPUT_DIR, \
    RENAL_TERMS_FILENAME, AL_SUBTYPES
from utils import parse_pipe_separated

SEARCH_TERMS = ['haematuria', 'hematuria', 'proteinuria']


def load_al_amyloidosis_patients():
    df = pd.read_excel(CLASSIFICATION_FILE, sheet_name='NEW_Amyloid_Patients')
    al_patients = df[df['Classified_Subtype'].isin(AL_SUBTYPES)]
    patient_ids = al_patients['Patient_ID'].astype(str).tolist()

    for subtype in AL_SUBTYPES:
        print(f"  {subtype}: {(al_patients['Classified_Subtype'] == subtype).sum()} patients")
    print(f"Total AL amyloidosis patients: {len(patient_ids)}")

    return patient_ids


def extract_renal_terms(patient_ids):
    visits_df = pd.read_excel(IC_EXCEL_FILE, sheet_name='Amyloid_449_Visit_IC_ENTIRE')
    visits_df['Patient_ID'] = visits_df['Patient_ID'].astype(str)
    al_visits = visits_df[visits_df['Patient_ID'].isin(patient_ids)]
    print(f"Amyloidosis visits: {len(visits_df):,} total, {len(al_visits):,} from AL patients")

    extracted = []
    for _, row in al_visits.iterrows():
        terms = parse_pipe_separated(row['SNOMED_Descriptions'])
        matching_terms = [t for t in terms if any(s in t.lower() for s in SEARCH_TERMS)]

        if matching_terms:
            extracted.append({
                'Patient_ID': row['Patient_ID'],
                'Visit_ID': row['Visit_ID'],
                'Visit_Date': row['Visit_Date'],
                'First_Amyloid_Date': row['First_Amyloid_Date'],
                'Months_From_Diagnosis': row['Months_From_Diagnosis'],
                'Matching_SNOMED_Terms': ' | '.join(matching_terms),
                'All_SNOMED_Terms': row['SNOMED_Descriptions'],
            })

    result_df = pd.DataFrame(extracted)
    if result_df.empty:
        print("No visits with renal terms found")
        return result_df

    result_df = result_df.sort_values(['Patient_ID', 'Months_From_Diagnosis'])

    n_haematuria = result_df['Matching_SNOMED_Terms'].str.contains('haematuria', case=False, na=False).sum()
    n_hematuria = result_df['Matching_SNOMED_Terms'].str.contains('hematuria', case=False, na=False).sum()
    n_proteinuria = result_df['Matching_SNOMED_Terms'].str.contains('proteinuria', case=False, na=False).sum()

    print(f"Visits with renal terms: {len(result_df)} ({result_df['Patient_ID'].nunique()} patients)")
    print(f"  haematuria: {n_haematuria}, hematuria: {n_hematuria}, proteinuria: {n_proteinuria}")

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
        result_df.to_excel(writer, sheet_name='Renal_Terms_Visits', index=False)

        patient_summary = result_df.groupby('Patient_ID').agg(
            Total_Visits_With_Renal_Terms=('Visit_ID', 'count'),
            First_Occurrence_Months=('Months_From_Diagnosis', 'min'),
            Last_Occurrence_Months=('Months_From_Diagnosis', 'max'),
            All_Unique_Renal_Terms=('Matching_SNOMED_Terms', lambda x: ' || '.join(x.unique())),
        ).reset_index()
        patient_summary.to_excel(writer, sheet_name='Patient_Summary', index=False)

        all_terms = [t for terms in result_df['Matching_SNOMED_Terms'] for t in parse_pipe_separated(terms)]
        term_counts = pd.Series(all_terms).value_counts().reset_index()
        term_counts.columns = ['SNOMED_Term', 'Frequency']
        term_counts.to_excel(writer, sheet_name='Term_Frequency', index=False)

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
    patient_ids = load_al_amyloidosis_patients()
    result_df = extract_renal_terms(patient_ids)

    output_file = os.path.join(OUTPUT_DIR, RENAL_TERMS_FILENAME)
    export_results(result_df, output_file)


if __name__ == "__main__":
    main()
