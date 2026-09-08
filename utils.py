import json
import re
import os
import pickle
from datetime import datetime
import pandas as pd


def load_json_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except OSError as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def has_amyloid_diagnosis(description):
    return bool(re.search(r'amyloid', description, re.IGNORECASE))


def calculate_age_from_dob(dob_str, reference_date=None):
    if not dob_str or dob_str == 'N/A':
        return None

    if reference_date is None:
        reference_date = datetime.now()

    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%Y-%m-%d %H:%M:%S.%f',
    ]

    for fmt in date_formats:
        try:
            birth_date = datetime.strptime(dob_str.strip(), fmt)
        except ValueError:
            continue

        age = reference_date.year - birth_date.year
        if reference_date.month < birth_date.month or \
           (reference_date.month == birth_date.month and reference_date.day < birth_date.day):
            age -= 1

        return age if 0 <= age <= 120 else None

    if ' ' in dob_str:
        return calculate_age_from_dob(dob_str.split(' ')[0], reference_date)

    return None


def standardize_gender(gender_raw):
    gender_raw = (gender_raw or '').upper()
    if gender_raw in ('M', 'MALE'):
        return 'M'
    if gender_raw in ('F', 'FEMALE'):
        return 'F'
    return None


def count_diagnosis_containing_visits(patient_details):
    """Counts visits with at least one non-empty diagnosis code."""
    if not isinstance(patient_details, dict):
        return 0
    return sum(
        1 for visit in patient_details.get('visits', {}).values()
        if visit.get('codes', [])
    )


def save_to_cache(cache_dir, cache_key, data, data_type):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}_{data_type}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except OSError as e:
        print(f"Failed to cache {data_type}: {e}")
        return False


def load_from_cache(cache_dir, cache_key, data_type):
    cache_file = os.path.join(cache_dir, f"{cache_key}_{data_type}.pkl")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError) as e:
        print(f"Failed to load cached {data_type}: {e}")
        return None


def parse_pipe_separated(value_string):
    """Parses a pipe-separated string into a list of stripped substrings.
    Used for IC values, SNOMED codes, and SNOMED descriptions alike."""
    if pd.isna(value_string) or not str(value_string).strip():
        return []
    return [v.strip() for v in str(value_string).split('|') if v.strip()]


def parse_ic_values(ic_string):
    return [float(v) for v in parse_pipe_separated(ic_string)]


def filter_out_amyloid_terms(ic_values, snomed_descriptions):
    """Removes IC values whose corresponding SNOMED description mentions
    amyloid/amyloidosis. Returns (filtered_ic_values, n_filtered)."""
    if not ic_values or not snomed_descriptions or len(ic_values) != len(snomed_descriptions):
        return ic_values, 0

    filtered = [(ic, desc) for ic, desc in zip(ic_values, snomed_descriptions)
                if not has_amyloid_diagnosis(desc)]
    filtered_ic = [ic for ic, _ in filtered]
    return filtered_ic, len(ic_values) - len(filtered_ic)


def get_significance_asterisks(p_value):
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"

