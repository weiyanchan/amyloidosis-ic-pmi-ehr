import os
from collections import defaultdict

from config import PATIENT_DATA_FILE, OUTPUT_DIR
from utils import load_json_file, has_amyloid_diagnosis, calculate_age_from_dob, \
    standardize_gender, count_diagnosis_containing_visits

MIN_DIAGNOSIS_VISITS = 2
CONTROL_RATIO = 100


def extract_flowchart_numbers(patient_data):
    """Tallies patient counts and exclusion reasons for the amyloidosis and
    control selection pathways (used for the manuscript's flow diagram)."""

    total_patients = len(patient_data)

    amyloid = defaultdict(int)
    controls = defaultdict(int)

    for patient_id, patient_details in patient_data.items():
        if not isinstance(patient_details, dict):
            continue

        visits = patient_details.get('visits', {})
        age = calculate_age_from_dob(patient_details.get('dob'))
        gender = standardize_gender(patient_details.get('gender'))

        has_amyloid = any(
            has_amyloid_diagnosis(code.get('description', ''))
            for visit in visits.values()
            for code in visit.get('codes', [])
        )

        diagnosis_visits = count_diagnosis_containing_visits(patient_details)
        has_only_empty_codes = len(visits) > 0 and diagnosis_visits == 0

        if has_amyloid:
            amyloid['total_found'] += 1

            if has_only_empty_codes:
                amyloid['excluded_empty_codes'] += 1
            elif age is None or gender is None:
                amyloid['excluded_missing_age_gender'] += 1
            elif not (18 <= age <= 100):
                amyloid['excluded_age_out_of_range'] += 1
            else:
                amyloid['final'] += 1

        else:
            controls['total_potential'] += 1

            if has_only_empty_codes:
                controls['excluded_empty_codes'] += 1
            elif diagnosis_visits < MIN_DIAGNOSIS_VISITS:
                controls['excluded_insufficient_visits'] += 1
            elif age is None or gender is None:
                controls['excluded_missing_age_gender'] += 1
            elif not (18 <= age <= 100):
                controls['excluded_age_out_of_range'] += 1
            else:
                controls['final_eligible'] += 1

    controls['final_matched'] = amyloid['final'] * CONTROL_RATIO

    print(f"Total patients: {total_patients:,}")
    print(f"Amyloidosis: {amyloid['total_found']:,} identified -> {amyloid['final']:,} final "
          f"(excluded: empty codes {amyloid['excluded_empty_codes']:,}, "
          f"missing age/gender {amyloid['excluded_missing_age_gender']:,}, "
          f"age out of range {amyloid['excluded_age_out_of_range']:,})")
    print(f"Controls: {controls['total_potential']:,} potential -> "
          f"{controls['final_eligible']:,} eligible -> {controls['final_matched']:,} matched (1:{CONTROL_RATIO}) "
          f"(excluded: empty codes {controls['excluded_empty_codes']:,}, "
          f"<{MIN_DIAGNOSIS_VISITS} diagnosis visits {controls['excluded_insufficient_visits']:,}, "
          f"missing age/gender {controls['excluded_missing_age_gender']:,}, "
          f"age out of range {controls['excluded_age_out_of_range']:,})")

    return {
        'total_patients': total_patients,
        'amyloid': dict(amyloid),
        'controls': dict(controls),
    }


def write_summary(numbers, output_file):
    with open(output_file, 'w') as f:
        f.write("Patient selection flowchart numbers\n\n")

        f.write(f"1. Starting population: {numbers['total_patients']:,}\n\n")

        a = numbers['amyloid']
        f.write("2. Amyloidosis pathway:\n")
        f.write(f"   Total identified: {a['total_found']:,}\n")
        f.write(f"   Excluded (empty codes only): {a['excluded_empty_codes']:,}\n")
        f.write(f"   Excluded (missing age/gender): {a['excluded_missing_age_gender']:,}\n")
        f.write(f"   Excluded (age out of range): {a['excluded_age_out_of_range']:,}\n")
        f.write(f"   Final: {a['final']:,}\n\n")

        c = numbers['controls']
        f.write("3. Control pathway:\n")
        f.write(f"   Total potential: {c['total_potential']:,}\n")
        f.write(f"   Excluded (empty codes only): {c['excluded_empty_codes']:,}\n")
        f.write(f"   Excluded (<{MIN_DIAGNOSIS_VISITS} diagnosis visits): {c['excluded_insufficient_visits']:,}\n")
        f.write(f"   Excluded (missing age/gender): {c['excluded_missing_age_gender']:,}\n")
        f.write(f"   Excluded (age out of range): {c['excluded_age_out_of_range']:,}\n")
        f.write(f"   Eligible: {c['final_eligible']:,}\n")
        f.write(f"   Final matched (1:{CONTROL_RATIO}): {c['final_matched']:,}\n")


def main():
    patient_data = load_json_file(PATIENT_DATA_FILE)
    if not patient_data:
        print("Failed to load patient data")
        return None

    numbers = extract_flowchart_numbers(patient_data)

    output_file = os.path.join(OUTPUT_DIR, "patient_selection_flowchart_numbers.txt")
    write_summary(numbers, output_file)
    print(f"Summary written to {output_file}")

    return numbers


if __name__ == "__main__":
    main()
