"""
File paths and shared parameters for the amyloidosis EHR analysis pipeline.
Update these to match your local data locations before running any script.
"""

import os

# Raw patient/lab data (matching script)
PATIENT_DATA_FILE = "data/patient_data.json"
LAB_DATA_FILES = [
    f"data/lab_splits/labs_{i}.json" for i in range(1, 8)
]

CACHE_DIR = "cache"

# Matching parameters
CONTROL_RATIO = 100
RANDOM_SEED = 42
MIN_DIAGNOSIS_VISITS = 2
VARIANCE_TOLERANCE = 0.15

# IC analysis
IC_EXCEL_FILE = "data/variance_matched_amyloid_ic_diagnosis_entire_dataset.xlsx"
CLASSIFICATION_FILE = "data/classification_amyloid_patients.xlsx"
LAB_FILE = "data/lab_test_extraction_new_classification.xlsx"

# Lab-derived finding thresholds (Methods)
PROTEINURIA_THRESHOLD = 0.15          # g/day
NEPHROTIC_THRESHOLD = 3.5             # g/day
MICROSCOPIC_HAEMATURIA_THRESHOLD = 3  # RBC/HPF (>=)

# AL amyloidosis subtype labels (used across renal-findings extraction scripts)
AL_SUBTYPES = ['AL amyloidosis', 'ATTRwt-Likely AL', 'Likely AL']

# Figure 5: combined renal + chest pain timeline
# RENAL_TERMS_FILENAME, LAB_RENAL_FINDINGS_FILENAME, and
# COMBINED_RENAL_FILENAME are pipeline-generated artifacts (written to and
# read from OUTPUT_DIR).
RENAL_TERMS_FILENAME = "al_amyloidosis_renal_terms_extraction.xlsx"
LAB_RENAL_FINDINGS_FILENAME = "al_amyloidosis_lab_derived_renal_findings_extraction.xlsx"
COMBINED_RENAL_FILENAME = "al_amyloidosis_combined_renal_findings.xlsx"

OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
