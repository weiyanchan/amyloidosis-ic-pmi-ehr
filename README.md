# amyloidosis-ic-pmi-ehr

This repository contains the analysis pipeline used to evaluate information
content (IC) and pointwise mutual information (PMI) as tools for identifying
prodromal clinical patterns preceding amyloidosis diagnosis in structured
EHR data.

## Requirements

- Python 3.9+
- pandas, numpy, scipy, statsmodels, matplotlib, seaborn, openpyxl

```
pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl
```

## Repository structure and pipeline order

Scripts are listed in the order they must be run, since each stage depends
on cached output from the one before it. All scripts share `config.py`
(file paths and parameters) and `utils.py` (shared helper functions).

| Order | Script | Produces | Manuscript reference |
|---|---|---|---|
| 1 | `matching.py` | Amyloidosis cohort + 1:100 variance-matched controls (cached) | Methods: Patient selection |
| 2 | `patient_selection_flowchart.py` | Cohort selection counts | Flow diagram |
| 3 | `compute_diagnosis_ic.py` | SNOMED-CT term IC values + visit-level IC records (cached, exported to Excel) | Methods: Information content; Equation 1 |
| 4 | `patient_ic_prediag_vs_prereference.py` | Patient-level IC, AL pre-diagnosis vs. control pre-reference | Figure 2 |
| 5 | `subgroup_ic_analysis.py` | Patient-level IC by subtype (Controls / Localised Cutaneous / AL incl. Likely AL / ATTR / Other) | Figure 2 subgroup panels |
| 6 | `temporal_ic_calendar_month.py` | Visit-level IC trend by calendar month | Figure 3A |
| 7 | `temporal_ic_visit_sequence.py` | Visit-level IC trend by visit sequence | Figure 3B |
| 8 | `pmi_analysis.py` | 2-term and 3-term PMI, SNOMED-only and lab-augmented, all subtypes vs. controls | Figure 4A-C |
| 9 | `pmi_effect_estimates.py` | Counts, percentages, PMI difference, FDR-corrected p, bootstrap stability for every tested pair/triplet | Supplementary Tables (reviewer-requested effect sizes/stability) |
| 10 | `extract_renal_terms.py` | AL amyloidosis visits with SNOMED-coded renal terms | Input to Figure 5 |
| 11 | `extract_lab_renal_findings.py` | AL amyloidosis visits with lab-derived renal findings | Input to Figure 5 |
| 12 | `merge_renal_findings.py` | Combined SNOMED + lab-derived renal findings per visit | Input to Figure 5 |
| 13 | `figure5_renal_chest_pain_timeline.py` | Pre-diagnosis renal + chest pain finding timeline, AL patients | Figure 5 |

`temporal_ic_common.py` holds logic shared by scripts 6 and 7 (not run directly).
Steps 10 and 11 are independent of each other and can be run in either order,
but both must run before step 12.

## Data availability

Patient-level EHR data were extracted from SingHealth institutions under
institutional data governance and are not publicly available due to patient
privacy restrictions. Data may be made available from the corresponding
author upon reasonable request, subject to institutional and ethics
approval.

To run this code against equivalent data, inputs should be structured as
follows:

- **Patient/lab JSON files** (`PATIENT_DATA_FILE`, `LAB_DATA_FILES` in
  `config.py`): one JSON object per patient, with `dob`, `gender`, `race`,
  and a `visits` dict; each visit has a `date` and a `codes` list of
  `{code, description}` diagnosis entries. Lab files are structured
  similarly, with `visits` containing `items` (`test`, value fields).
- **Classification file** (`CLASSIFICATION_FILE`): one row per amyloidosis
  patient, columns `Patient_ID`, `Classified_Subtype` (values: `AL
  amyloidosis`, `ATTRv`, `ATTRwt`, `ATTR`, `CAA`, `Gelsolin`, `Likely AL`,
  `ATTRwt-Likely AL`, `Skin`, `Others`).
- **Lab extraction file** (`LAB_FILE`): sheets `AL_Amyloidosis`,
  `ATTR_Amyloidosis`, `Controls`, each with `Visit_ID` and lab fields
  (`TPU`, `PU`, `P24U`, `URBC`).

## Lab-derived term definitions (Methods)

- Proteinuria: urine protein concentration > 0.15 g/day
- Nephrotic-range proteinuria: urine protein concentration > 3.5 g/day
- Microscopic haematuria: urine red blood cell count >= 3 RBC/HPF

These thresholds are defined once, in `config.py`
(`PROTEINURIA_THRESHOLD`, `NEPHROTIC_THRESHOLD`,
`MICROSCOPIC_HAEMATURIA_THRESHOLD`), and used identically by
`pmi_analysis.py` and `extract_lab_renal_findings.py`.

## Reproducibility notes

- All cached intermediate data (matched controls, computed IC values) is
  keyed by a content hash of the input data files (see
  `matching.get_cache_key`), computed fresh each run rather than hardcoded
  -- reruns against updated or different data will not silently reuse a
  stale cache under the wrong key.
- The minimum-patient threshold for PMI pairs/triplets (>=2 patients in
  both groups) is defined once (`MIN_PATIENTS_THRESHOLD`) and applied
  consistently across `pmi_analysis.py` and `pmi_effect_estimates.py`.
