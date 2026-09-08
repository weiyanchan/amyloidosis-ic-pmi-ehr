import pandas as pd
import random
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import gc
import os
import hashlib
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp
import warnings
warnings.filterwarnings('ignore')

from config import (
    PATIENT_DATA_FILE, LAB_DATA_FILES, CACHE_DIR, OUTPUT_DIR,
    CONTROL_RATIO, RANDOM_SEED, MIN_DIAGNOSIS_VISITS, VARIANCE_TOLERANCE,
)
from utils import load_json_file, has_amyloid_diagnosis, calculate_age_from_dob, \
    standardize_gender, save_to_cache as _save_to_cache, load_from_cache as _load_from_cache

CACHE_VERSION = "v6"


def get_file_hash(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]
    except OSError:
        return "unknown"


def get_cache_key(patient_data_file, lab_data_files):
    file_hashes = [get_file_hash(patient_data_file)]
    file_hashes += [get_file_hash(f) for f in lab_data_files]
    combined_hash = hashlib.md5(''.join(file_hashes).encode()).hexdigest()[:16]
    return f"{CACHE_VERSION}_{combined_hash}"


def save_to_cache(cache_key, data, data_type):
    return _save_to_cache(CACHE_DIR, cache_key, data, data_type)


def load_from_cache(cache_key, data_type):
    return _load_from_cache(CACHE_DIR, cache_key, data_type)


def load_all_lab_data(lab_data_files):
    all_lab_data = {}
    for lab_file in lab_data_files:
        lab_data = load_json_file(lab_file)
        if lab_data:
            all_lab_data.update(lab_data)
            del lab_data
            gc.collect()

    print(f"Loaded lab data for {len(all_lab_data)} patients")
    return all_lab_data


def count_patient_visits_detailed(patient_id, patient_details, all_lab_data):
    """Counts diagnosis-only, labs-only, and both-type visits for a patient.
    Only visits with a non-empty diagnosis code list are counted as diagnosis
    visits."""

    if not isinstance(patient_details, dict):
        return {'diagnosis_only': 0, 'labs_only': 0, 'both': 0, 'total': 0}

    diagnosis_visits = {
        visit_id for visit_id, visit_data in patient_details.get('visits', {}).items()
        if visit_data.get('codes', [])
    }

    lab_visits = set()
    patient_lab_data = all_lab_data.get(patient_id)
    if isinstance(patient_lab_data, dict) and 'visits' in patient_lab_data:
        lab_visits = set(patient_lab_data['visits'].keys())

    both_visits = diagnosis_visits & lab_visits
    diagnosis_only_visits = diagnosis_visits - lab_visits
    labs_only_visits = lab_visits - diagnosis_visits

    return {
        'diagnosis_only': len(diagnosis_only_visits),
        'labs_only': len(labs_only_visits),
        'both': len(both_visits),
        'total': len(diagnosis_visits | lab_visits),
    }


def extract_all_amyloid_and_filtered_controls(patient_data, all_lab_data, min_diagnosis_visits=2):
    """Returns all amyloid patients (unfiltered) and controls with at least
    min_diagnosis_visits diagnosis-containing visits, valid age (18-100), and
    valid gender."""

    amyloid_patients = {}
    eligible_controls = []

    amyloid_no_age_gender = 0
    controls_filtered_no_diag = 0
    controls_filtered_no_age_gender = 0

    for patient_id, patient_details in patient_data.items():
        if not isinstance(patient_details, dict):
            continue

        dob = patient_details.get('dob')
        gender = standardize_gender(patient_details.get('gender'))
        age = calculate_age_from_dob(dob) if dob else None
        race = patient_details.get('race', '')

        visit_details = count_patient_visits_detailed(patient_id, patient_details, all_lab_data)
        diagnosis_containing_visits = visit_details['diagnosis_only'] + visit_details['both']

        has_amyloid = False
        amyloid_diagnoses = []
        for visit_data in patient_details.get('visits', {}).values():
            for code in visit_data.get('codes', []):
                description = code.get('description', '')
                if has_amyloid_diagnosis(description):
                    has_amyloid = True
                    amyloid_diagnoses.append(description)

        patient_summary = {
            'Patient_ID': patient_id,
            'Date_of_Birth': dob,
            'Age': age,
            'Gender': gender,
            'Race': race,
            'Total_Visits': visit_details['total'],
            'Diagnosis_Only_Visits': visit_details['diagnosis_only'],
            'Labs_Only_Visits': visit_details['labs_only'],
            'Both_Visits': visit_details['both'],
            'Diagnosis_Containing_Visits': diagnosis_containing_visits,
            'Has_Amyloid': has_amyloid,
            'Amyloid_Diagnoses': amyloid_diagnoses if has_amyloid else [],
        }

        if has_amyloid:
            amyloid_patients[patient_id] = patient_summary
            if age is None or gender is None:
                amyloid_no_age_gender += 1
            continue

        if diagnosis_containing_visits < min_diagnosis_visits:
            controls_filtered_no_diag += 1
            continue

        if age is None or gender is None or not (18 <= age <= 100):
            controls_filtered_no_age_gender += 1
            continue

        eligible_controls.append(patient_summary)

    print(f"Amyloid patients: {len(amyloid_patients)} "
          f"({amyloid_no_age_gender} without valid age/gender)")
    print(f"Eligible controls: {len(eligible_controls)}")
    print(f"Controls excluded - insufficient diagnosis visits: {controls_filtered_no_diag:,}, "
          f"missing age/gender: {controls_filtered_no_age_gender:,}")

    return amyloid_patients, eligible_controls


def variance_aware_age_gender_matching(amyloid_patients, eligible_controls,
                                        control_ratio=100, random_seed=42,
                                        variance_tolerance=0.15):
    """Stratified age/gender matching, followed by iterative adjustment so
    the matched controls' age SD falls within variance_tolerance of the
    amyloid cohort's age SD (needed to pass a K-S test on the age
    distribution, not just the mean)."""

    random.seed(random_seed)
    np.random.seed(random_seed)

    amyloid_list = list(amyloid_patients.values())
    target_ages = [p['Age'] for p in amyloid_list if p['Age'] is not None]
    target_genders = [p['Gender'] for p in amyloid_list if p['Gender'] is not None]

    if not target_ages:
        return []

    target_mean = np.mean(target_ages)
    target_std = np.std(target_ages, ddof=1)

    gender_counts = Counter(target_genders)
    gender_proportions = {k: v / sum(gender_counts.values()) for k, v in gender_counts.items()}

    print(f"Target age: {target_mean:.1f} +/- {target_std:.1f} years, gender: {dict(gender_proportions)}")

    # Stratify by age decile and gender
    age_percentiles = range(0, 101, 10)
    age_bins = sorted(set(int(np.percentile(target_ages, p)) for p in age_percentiles)) + [120]

    def assign_bin(age):
        for i in range(len(age_bins) - 1):
            if age <= age_bins[i + 1]:
                return i
        return len(age_bins) - 1

    strata = defaultdict(list)
    for control in eligible_controls:
        if control['Age'] is None or control['Gender'] not in gender_proportions:
            continue
        strata[(assign_bin(control['Age']), control['Gender'])].append(control)

    amyloid_strata_counts = defaultdict(int)
    for patient in amyloid_list:
        if patient['Age'] is None or patient['Gender'] is None:
            continue
        amyloid_strata_counts[(assign_bin(patient['Age']), patient['Gender'])] += 1

    total_amyloid_in_strata = sum(amyloid_strata_counts.values())
    total_controls_needed = len(amyloid_patients) * control_ratio

    selected_controls = []
    for stratum_key, target_count in amyloid_strata_counts.items():
        available = strata.get(stratum_key, [])
        needed = int(total_controls_needed * (target_count / total_amyloid_in_strata))

        if needed > 0 and len(available) >= needed:
            selected_controls.extend(random.sample(available, needed))
        elif available:
            selected_controls.extend(available)

    if len(selected_controls) < total_controls_needed:
        selected_ids = {p['Patient_ID'] for p in selected_controls}
        remaining = [p for p in eligible_controls if p['Patient_ID'] not in selected_ids]
        shortfall = total_controls_needed - len(selected_controls)
        if remaining:
            selected_controls.extend(random.sample(remaining, min(shortfall, len(remaining))))

    if len(selected_controls) > total_controls_needed:
        selected_controls = random.sample(selected_controls, total_controls_needed)

    print(f"Initial stratified sample: {len(selected_controls):,} controls")

    # Adjust variance to fall within tolerance of the target SD
    min_std, max_std = target_std * (1 - variance_tolerance), target_std * (1 + variance_tolerance)
    current_std = np.std([p['Age'] for p in selected_controls if p['Age'] is not None], ddof=1)

    iteration = 0
    while (current_std < min_std or current_std > max_std) and iteration < 100:
        iteration += 1
        selected_ids = {p['Patient_ID'] for p in selected_controls}

        if current_std > max_std:
            # too spread out: swap the most extreme-age controls for ones nearer the mean
            candidates = [p for p in eligible_controls
                          if p['Patient_ID'] not in selected_ids and p['Age'] is not None
                          and abs(p['Age'] - target_mean) < target_std * 1.5]
            to_remove = sorted(selected_controls, key=lambda p: abs(p['Age'] - target_mean), reverse=True)
        else:
            # too narrow: swap central controls for ones further from the mean
            candidates = [p for p in eligible_controls
                          if p['Patient_ID'] not in selected_ids and p['Age'] is not None
                          and abs(p['Age'] - target_mean) > target_std * 0.8]
            to_remove = sorted(selected_controls, key=lambda p: abs(p['Age'] - target_mean))

        if not candidates:
            break

        n_swap = max(1, int(len(selected_controls) * 0.05))
        replacements = random.sample(candidates, min(n_swap, len(candidates)))
        for old in to_remove[:len(replacements)]:
            selected_controls.remove(old)
        selected_controls.extend(replacements)

        current_std = np.std([p['Age'] for p in selected_controls if p['Age'] is not None], ddof=1)

    final_ages = [p['Age'] for p in selected_controls if p['Age'] is not None]
    print(f"Final controls (after {iteration} variance-adjustment iterations): "
          f"{np.mean(final_ages):.1f} +/- {np.std(final_ages, ddof=1):.1f} years "
          f"(target {target_mean:.1f} +/- {target_std:.1f})")

    return selected_controls


def comprehensive_statistical_tests(amyloid_patients, matched_controls, alpha=0.05):
    """Mann-Whitney/K-S tests on age, chi-square on gender, and Mann-Whitney/
    K-S on healthcare utilisation measures (visit counts), the latter
    reported but not used as a matching criterion."""

    results = {}
    amyloid_list = list(amyloid_patients.values())

    amy_ages = [p['Age'] for p in amyloid_list if p['Age'] is not None]
    ctrl_ages = [p['Age'] for p in matched_controls if p['Age'] is not None]

    if amy_ages and ctrl_ages:
        _, mw_p = mannwhitneyu(amy_ages, ctrl_ages, alternative='two-sided')
        _, ks_p = ks_2samp(amy_ages, ctrl_ages)
        pooled_std = np.sqrt((np.var(amy_ages) + np.var(ctrl_ages)) / 2)
        cohens_d = abs(np.mean(amy_ages) - np.mean(ctrl_ages)) / pooled_std if pooled_std > 0 else 0

        results['age'] = {
            'mann_whitney_p': mw_p, 'kolmogorov_smirnov_p': ks_p, 'cohens_d': cohens_d,
            'amy_mean': np.mean(amy_ages), 'ctrl_mean': np.mean(ctrl_ages),
            'amy_std': np.std(amy_ages), 'ctrl_std': np.std(ctrl_ages),
        }
        print(f"Age: amyloid {np.mean(amy_ages):.1f}+/-{np.std(amy_ages):.1f} (n={len(amy_ages)}), "
              f"controls {np.mean(ctrl_ages):.1f}+/-{np.std(ctrl_ages):.1f} (n={len(ctrl_ages)}); "
              f"MW p={mw_p:.4f}, KS p={ks_p:.4f}, Cohen's d={cohens_d:.3f}")

    amy_genders = [p['Gender'] for p in amyloid_list if p['Gender'] is not None]
    ctrl_genders = [p['Gender'] for p in matched_controls if p['Gender'] is not None]

    if amy_genders and ctrl_genders:
        amy_counts_s = pd.Series(amy_genders).value_counts().sort_index()
        ctrl_counts_s = pd.Series(ctrl_genders).value_counts().sort_index()
        all_genders = sorted(set(amy_counts_s.index) | set(ctrl_counts_s.index))
        amy_counts = [amy_counts_s.get(g, 0) for g in all_genders]
        ctrl_counts = [ctrl_counts_s.get(g, 0) for g in all_genders]

        if all(a + c > 0 for a, c in zip(amy_counts, ctrl_counts)):
            chi2_stat, chi2_p, _, _ = chi2_contingency([amy_counts, ctrl_counts])
            results['gender'] = {
                'chi_square_p': chi2_p, 'chi_square_stat': chi2_stat,
                'amy_distribution': dict(zip(all_genders, amy_counts)),
                'ctrl_distribution': dict(zip(all_genders, ctrl_counts)),
            }
            print(f"Gender: amyloid {dict(zip(all_genders, amy_counts))}, "
                  f"controls {dict(zip(all_genders, ctrl_counts))}; chi2 p={chi2_p:.4f}")

    visit_types = ['Total_Visits', 'Diagnosis_Only_Visits', 'Labs_Only_Visits',
                   'Both_Visits', 'Diagnosis_Containing_Visits']

    for visit_type in visit_types:
        amy_visits = [p[visit_type] for p in amyloid_list]
        ctrl_visits = [p[visit_type] for p in matched_controls]
        if not (amy_visits and ctrl_visits):
            continue

        _, mw_p = mannwhitneyu(amy_visits, ctrl_visits, alternative='two-sided')
        _, ks_p = ks_2samp(amy_visits, ctrl_visits)

        results[visit_type.lower()] = {
            'mann_whitney_p': mw_p, 'kolmogorov_smirnov_p': ks_p,
            'amy_median': np.median(amy_visits), 'ctrl_median': np.median(ctrl_visits),
            'amy_mean': np.mean(amy_visits), 'ctrl_mean': np.mean(ctrl_visits),
        }
        print(f"{visit_type} (not matched): amyloid median {np.median(amy_visits):.1f}, "
              f"controls median {np.median(ctrl_visits):.1f}, MW p={mw_p:.4f}")

    return results


def export_to_excel(results, filename=None):
    filename = filename or os.path.join(OUTPUT_DIR, "variance_matched_controls_1to100.xlsx")

    amyloid_df = pd.DataFrame.from_dict(results['amyloid_patients'], orient='index')
    amyloid_df['Patient_Type'] = 'Amyloid'
    amyloid_df = amyloid_df.reset_index(drop=True)

    controls_df = pd.DataFrame(results['matched_controls'])
    controls_df['Patient_Type'] = 'Control'
    controls_df['Has_Amyloid'] = False
    controls_df['Amyloid_Diagnoses'] = ''

    combined_df = pd.concat([amyloid_df, controls_df], ignore_index=True)

    test_summary = []
    for test_name, test_data in results['statistical_tests'].items():
        if isinstance(test_data, dict):
            test_summary.append({
                'Test_Variable': test_name.replace('_', ' ').title(),
                'Mann_Whitney_P': test_data.get('mann_whitney_p', 'N/A'),
                'Kolmogorov_Smirnov_P': test_data.get('kolmogorov_smirnov_p', 'N/A'),
                'Chi_Square_P': test_data.get('chi_square_p', 'N/A'),
                'Amyloid_Mean': test_data.get('amy_mean', test_data.get('amy_median', 'N/A')),
                'Control_Mean': test_data.get('ctrl_mean', test_data.get('ctrl_median', 'N/A')),
                'Effect_Size_Cohens_D': test_data.get('cohens_d', 'N/A'),
            })

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        combined_df.to_excel(writer, sheet_name='Combined_Data', index=False)
        amyloid_df.to_excel(writer, sheet_name='Amyloid_Patients', index=False)
        controls_df.to_excel(writer, sheet_name='Matched_Controls', index=False)
        pd.DataFrame(test_summary).to_excel(writer, sheet_name='Statistical_Tests', index=False)

    print(f"Exported {len(amyloid_df)} amyloid patients and {len(controls_df)} matched controls to {filename}")


def main():
    start_time = datetime.now()
    cache_key = get_cache_key(PATIENT_DATA_FILE, LAB_DATA_FILES)

    amyloid_patients = load_from_cache(cache_key, "all_amyloid_patients")
    eligible_controls = load_from_cache(cache_key, "eligible_controls")

    if amyloid_patients is None or eligible_controls is None:
        patient_data = load_json_file(PATIENT_DATA_FILE)
        all_lab_data = load_all_lab_data(LAB_DATA_FILES)

        amyloid_patients, eligible_controls = extract_all_amyloid_and_filtered_controls(
            patient_data, all_lab_data, min_diagnosis_visits=MIN_DIAGNOSIS_VISITS
        )

        save_to_cache(cache_key, amyloid_patients, "all_amyloid_patients")
        save_to_cache(cache_key, eligible_controls, "eligible_controls")

    matched_controls = variance_aware_age_gender_matching(
        amyloid_patients, eligible_controls,
        control_ratio=CONTROL_RATIO, random_seed=RANDOM_SEED,
        variance_tolerance=VARIANCE_TOLERANCE,
    )

    test_results = comprehensive_statistical_tests(amyloid_patients, matched_controls)

    final_results = {
        'amyloid_patients': amyloid_patients,
        'matched_controls': matched_controls,
        'statistical_tests': test_results,
        'metadata': {
            'num_amyloid_patients': len(amyloid_patients),
            'num_matched_controls': len(matched_controls),
            'matching_method': 'variance_aware_age_gender',
            'variance_tolerance': VARIANCE_TOLERANCE,
            'timestamp': datetime.now().isoformat(),
        },
    }

    save_to_cache(cache_key, matched_controls, "matched_controls")
    export_to_excel(final_results)

    print(f"Completed in {(datetime.now() - start_time).total_seconds():.1f}s")
    return final_results


if __name__ == "__main__":
    main()
