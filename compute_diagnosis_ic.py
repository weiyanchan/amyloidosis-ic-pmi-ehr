"""
Computes Information Content (IC) for every SNOMED-CT diagnosis term found
among the amyloidosis and matched-control cohorts, using term frequency
across the entire ~1.28M patient dataset as the probability denominator
(Equation 1 in the manuscript). Produces visit-level IC records for both
cohorts and caches them for downstream subgroup analysis.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
from collections import defaultdict

from config import PATIENT_DATA_FILE, LAB_DATA_FILES, CACHE_DIR, OUTPUT_DIR
from matching import get_cache_key
from utils import load_json_file, has_amyloid_diagnosis, save_to_cache, load_from_cache


def load_reference_patient_ids(cache_key):
    amyloid_patients = load_from_cache(CACHE_DIR, cache_key, "all_amyloid_patients")
    matched_controls = load_from_cache(CACHE_DIR, cache_key, "matched_controls")

    if amyloid_patients is None or matched_controls is None:
        raise FileNotFoundError(
            "Amyloid/matched control cache not found. Run matching.py first."
        )

    amyloid_ids = set(amyloid_patients.keys())
    control_ids = {c['Patient_ID'] for c in matched_controls}

    print(f"Amyloid patients: {len(amyloid_ids):,}")
    print(f"Matched controls: {len(control_ids):,}")

    return amyloid_ids, control_ids


def parse_visit_date(date_str):
    if not date_str:
        return None
    if not isinstance(date_str, str):
        return date_str
    try:
        return datetime.strptime(date_str.strip(), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None


def extract_target_diagnoses(reference_patient_ids, patient_data):
    """All unique, non-empty diagnosis descriptions appearing in the
    amyloid + control reference population. These are the terms whose IC
    values will subsequently be estimated from the full dataset."""

    target_diagnoses = set()
    n_with_diagnoses = 0

    for patient_id in reference_patient_ids:
        patient_details = patient_data.get(patient_id)
        if not isinstance(patient_details, dict):
            continue

        has_diagnosis = False
        for visit in patient_details.get('visits', {}).values():
            for code in visit.get('codes', []):
                description = code.get('description', '').strip()
                if description:
                    target_diagnoses.add(description)
                    has_diagnosis = True

        n_with_diagnoses += has_diagnosis

    print(f"Reference population with >=1 diagnosis: {n_with_diagnoses:,}")
    print(f"Unique target diagnoses: {len(target_diagnoses):,}")

    return target_diagnoses


def count_diagnoses_across_dataset(target_diagnoses, patient_data):
    """Counts how often each target diagnosis occurs across every visit in
    the full dataset, plus the total number of diagnosis-containing visits
    (the denominator for IC, per Equation 1)."""

    diagnosis_counts = defaultdict(int)
    total_diagnosis_visits = 0
    total_instances = 0
    n_with_diagnoses = 0

    for patient_details in patient_data.values():
        if not isinstance(patient_details, dict):
            continue

        patient_has_diagnosis = False
        for visit in patient_details.get('visits', {}).values():
            codes = visit.get('codes', [])
            descriptions = [c.get('description', '').strip() for c in codes]
            descriptions = [d for d in descriptions if d]

            if not descriptions:
                continue

            total_diagnosis_visits += 1
            for description in descriptions:
                patient_has_diagnosis = True
                if description in target_diagnoses:
                    diagnosis_counts[description] += 1
                    total_instances += 1

        n_with_diagnoses += patient_has_diagnosis

    print(f"Total diagnosis-containing visits: {total_diagnosis_visits:,}")
    print(f"Target diagnosis instances: {total_instances:,}")
    print(f"Unique target diagnoses found: {len(diagnosis_counts):,}")

    return dict(diagnosis_counts), total_diagnosis_visits


def calculate_ic_values(diagnosis_counts, total_diagnosis_visits):
    """IC(term) = -log2(count / total_diagnosis_visits) -- Equation 1."""
    diagnosis_ic = {
        diagnosis: {
            'count': count,
            'probability': count / total_diagnosis_visits,
            'ic': -math.log2(count / total_diagnosis_visits),
        }
        for diagnosis, count in diagnosis_counts.items()
    }

    ic_values = [v['ic'] for v in diagnosis_ic.values()]
    print(f"IC range: {min(ic_values):.3f} (most common) to {max(ic_values):.3f} (rarest)")

    return diagnosis_ic


def find_reference_date(patient_id, patient_details, patient_type):
    """Amyloid patients: date of first amyloid-related diagnosis code.
    Controls: midpoint between first and last visit (see Methods)."""

    visits = patient_details.get('visits', {})

    if patient_type == "amyloid":
        amyloid_dates = [
            parse_visit_date(v.get('date')) for v in visits.values()
            if parse_visit_date(v.get('date')) and
            any(has_amyloid_diagnosis(c.get('description', '')) for c in v.get('codes', []))
        ]
        return min(amyloid_dates) if amyloid_dates else None

    visit_dates = sorted(d for d in (parse_visit_date(v.get('date')) for v in visits.values()) if d)
    if not visit_dates:
        return None
    return visit_dates[0] + (visit_dates[-1] - visit_dates[0]) / 2


def build_visit_level_records(patient_ids, patient_data, diagnosis_ic, patient_type):
    """One record per diagnosis-containing visit, with the visit's mean IC
    and its time offset from the patient's reference date."""

    records = []
    n_with_valid_reference = 0

    for patient_id in patient_ids:
        patient_details = patient_data.get(patient_id)
        if not isinstance(patient_details, dict):
            continue

        reference_date = find_reference_date(patient_id, patient_details, patient_type)
        if reference_date is None:
            continue
        n_with_valid_reference += 1

        for visit_id, visit in patient_details.get('visits', {}).items():
            visit_date = parse_visit_date(visit.get('date'))
            if visit_date is None:
                continue

            descriptions = [c.get('description', '').strip() for c in visit.get('codes', [])]
            descriptions = [d for d in descriptions if d]
            if not descriptions:
                continue

            ic_values = [diagnosis_ic[d]['ic'] for d in descriptions if d in diagnosis_ic]
            months_offset = round((visit_date - reference_date).days / 30.44, 2)

            record = {
                'Patient_ID': patient_id,
                'Visit_ID': visit_id,
                'Visit_Date': visit.get('date'),
                'Num_SNOMED_Terms': len(descriptions),
                'SNOMED_Descriptions': ' | '.join(descriptions),
                'Mean_IC': round(np.mean(ic_values), 6) if ic_values else None,
                'IC_Values': ' | '.join(f"{ic:.6f}" for ic in ic_values) if ic_values else None,
            }

            if patient_type == "amyloid":
                record['First_Amyloid_Date'] = reference_date.strftime('%Y-%m-%d %H:%M:%S')
                record['Months_From_Diagnosis'] = months_offset
            else:
                record['Reference_Timepoint'] = f"Midpoint: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}"
                record['Months_From_Reference'] = months_offset

            records.append(record)

    print(f"{patient_type}: {n_with_valid_reference}/{len(patient_ids)} patients with a valid reference date, "
          f"{len(records):,} visit records generated")

    return records


def export_results(diagnosis_ic, amyloid_visits, control_visits, total_diagnosis_visits,
                    total_patients, total_instances, n_controls, output_file):
    ic_rows = sorted(
        ({'Diagnosis_Description': d, 'Count_Across_All_Patients': v['count'],
          'Probability': round(v['probability'], 10), 'Information_Content': round(v['ic'], 6)}
         for d, v in diagnosis_ic.items()),
        key=lambda r: r['Information_Content'], reverse=True,
    )

    ic_df = pd.DataFrame(ic_rows)
    amyl_df = pd.DataFrame(amyloid_visits)
    control_df = pd.DataFrame(control_visits)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        ic_df.to_excel(writer, sheet_name='Diagnosis_IC_ENTIRE_DATASET', index=False)
        amyl_df.to_excel(writer, sheet_name='Amyloid_449_Visit_IC_ENTIRE', index=False)
        control_df.to_excel(writer, sheet_name='Control_NEW_VARIANCE_Visit', index=False)

        summary = pd.DataFrame([
            ['Total patients in dataset', total_patients],
            ['Total diagnosis-containing visits', total_diagnosis_visits],
            ['Total diagnosis instances', total_instances],
            ['Unique diagnoses', len(ic_df)],
            ['Amyloid patients', amyl_df['Patient_ID'].nunique() if len(amyl_df) else 0],
            ['Matched controls', n_controls],
            ['Mean IC (entire dataset)', round(ic_df['Information_Content'].mean(), 6)],
            ['Amyloid mean IC per visit', round(amyl_df['Mean_IC'].mean(), 6) if len(amyl_df) else None],
            ['Control mean IC per visit', round(control_df['Mean_IC'].mean(), 6) if len(control_df) else None],
        ])
        summary.to_excel(writer, sheet_name='Summary', index=False, header=False)

    print(f"Exported: {output_file}")
    return ic_df, amyl_df, control_df


def main():
    start_time = datetime.now()
    cache_key = get_cache_key(PATIENT_DATA_FILE, LAB_DATA_FILES)

    amyloid_ids, control_ids = load_reference_patient_ids(cache_key)
    patient_data = load_json_file(PATIENT_DATA_FILE)
    if not patient_data:
        raise RuntimeError("Failed to load patient data")
    print(f"Loaded {len(patient_data):,} patients")

    target_diagnoses = extract_target_diagnoses(amyloid_ids | control_ids, patient_data)
    diagnosis_counts, total_diagnosis_visits = count_diagnoses_across_dataset(target_diagnoses, patient_data)
    total_instances = sum(diagnosis_counts.values())
    diagnosis_ic = calculate_ic_values(diagnosis_counts, total_diagnosis_visits)

    amyloid_visits = build_visit_level_records(amyloid_ids, patient_data, diagnosis_ic, "amyloid")
    control_visits = build_visit_level_records(control_ids, patient_data, diagnosis_ic, "control")

    output_file = f"{OUTPUT_DIR}variance_matched_amyloid_ic_diagnosis_entire_dataset.xlsx"
    ic_df, amyl_df, control_df = export_results(
        diagnosis_ic, amyloid_visits, control_visits, total_diagnosis_visits,
        len(patient_data), total_instances, len(control_ids), output_file,
    )

    save_to_cache(CACHE_DIR, cache_key, {
        'diagnosis_ic': diagnosis_ic,
        'amyloid_visit_data': amyloid_visits,
        'control_visit_data': control_visits,
        'metadata': {
            'amyloid_patients': len(amyloid_ids),
            'control_patients': len(control_ids),
            'timestamp': datetime.now().isoformat(),
        },
    }, "diagnosis_ic_NEW_variance_matched")

    print(f"Completed in {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")

    return {'diagnosis_ic': diagnosis_ic, 'amyloid_visit_data': amyloid_visits,
            'control_visit_data': control_visits, 'df1': ic_df, 'df2': amyl_df, 'df3': control_df}


if __name__ == "__main__":
    main()
