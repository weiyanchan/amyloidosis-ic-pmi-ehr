"""
Supplementary evidence table for PMI co-occurrence findings (AL amyloidosis
vs. controls): counts, percentages, PMI difference (point estimate and
bootstrap 95% CI), and FDR-corrected p-value, for every pair and triplet
that passed the existing minimum-patient threshold in
compare_2pmi/compare_3pmi.

Requires pmi_analysis.py in the same directory (or on the Python path).

Output: one Excel workbook with four submission-ready sheets
(Pairs/Triplets x SNOMED-only/SNOMED-plus-labs), each combining the point
estimate and bootstrap columns in a single row per finding, sorted by
corrected p-value.
"""

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from pmi_analysis import (
    load_and_classify_patients, load_and_augment_data, aggregate_patient_terms,
    calculate_2term_pmi, calculate_3term_pmi, compare_2pmi, compare_3pmi,
)

MIN_PATIENTS_THRESHOLD = 2
SUBGROUP = "AL_Amyloidosis"
N_BOOTSTRAP = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Percentages (counts are already in the comparison table; PMI point
# estimates and pmi_difference are also already present via calculate_2/3term_pmi)
# ---------------------------------------------------------------------------

def add_percentages(df, count_col, total_col, ctrl_count_col, ctrl_total_col):
    """Adds % subgroup / % control columns to a compare_2pmi/compare_3pmi table.
    Returns df unchanged if empty (e.g. no pairs/triplets met the threshold)."""
    if df.empty or count_col not in df.columns:
        return df

    df = df.copy()
    df["pct_subgroup"] = 100 * df[count_col] / df[total_col]
    df["pct_control"] = 100 * df[ctrl_count_col] / df[ctrl_total_col]
    return df


# ---------------------------------------------------------------------------
# Bootstrap: resample both groups, recompute the PMI difference
# ---------------------------------------------------------------------------

def _presence_counts(terms_by_patient, sample_ids, term_names):
    n = len(sample_ids)
    has_term = {t: 0 for t in term_names}
    has_all = 0

    for pid in sample_ids:
        terms = terms_by_patient[pid]
        flags = [t in terms for t in term_names]
        for t, flag in zip(term_names, flags):
            has_term[t] += flag
        has_all += all(flags)

    return n, has_term, has_all


def bootstrap_pmi_difference(subgroup_terms, control_terms, term_names, n_boot, seed):
    """
    Resamples the subgroup and control cohorts independently (with
    replacement, group membership preserved) and recomputes the PMI
    difference between groups at each iteration.
    Returns an array of PMI-difference values, one per iteration (NaN where
    undefined, e.g. zero co-occurrence in a resample).
    """
    rng = np.random.default_rng(seed)
    sub_ids = np.array(list(subgroup_terms.keys()))
    ctrl_ids = np.array(list(control_terms.keys()))

    pmi_diffs = []
    for _ in range(n_boot):
        sub_sample = rng.choice(sub_ids, size=len(sub_ids), replace=True)
        ctrl_sample = rng.choice(ctrl_ids, size=len(ctrl_ids), replace=True)

        n_sub, sub_has, sub_all = _presence_counts(subgroup_terms, sub_sample, term_names)
        n_ctrl, ctrl_has, ctrl_all = _presence_counts(control_terms, ctrl_sample, term_names)

        p_sub = [sub_has[t] / n_sub for t in term_names]
        p_ctrl = [ctrl_has[t] / n_ctrl for t in term_names]
        p_sub_joint = sub_all / n_sub
        p_ctrl_joint = ctrl_all / n_ctrl

        if p_sub_joint > 0 and all(p > 0 for p in p_sub) and \
           p_ctrl_joint > 0 and all(p > 0 for p in p_ctrl):
            pmi_sub = np.log2(p_sub_joint / np.prod(p_sub))
            pmi_ctrl = np.log2(p_ctrl_joint / np.prod(p_ctrl))
            pmi_diffs.append(pmi_sub - pmi_ctrl)
        else:
            pmi_diffs.append(np.nan)

    return np.array(pmi_diffs)


def summarize_bootstrap(pmi_diffs):
    valid = pmi_diffs[~np.isnan(pmi_diffs)]
    n_total = len(pmi_diffs)

    if len(valid) == 0:
        return {
            "boot_pmi_diff_median": np.nan,
            "boot_pmi_diff_ci_low": np.nan,
            "boot_pmi_diff_ci_high": np.nan,
            "boot_pct_pmi_diff_positive": np.nan,
            "boot_pct_undefined": 100.0,
        }

    return {
        "boot_pmi_diff_median": np.median(valid),
        "boot_pmi_diff_ci_low": np.percentile(valid, 2.5),
        "boot_pmi_diff_ci_high": np.percentile(valid, 97.5),
        "boot_pct_pmi_diff_positive": 100 * (valid > 0).mean(),
        "boot_pct_undefined": 100 * (n_total - len(valid)) / n_total,
    }


def add_bootstrap_columns(df, term_col, subgroup_terms, control_terms, n_terms, n_boot, seed):
    """
    Runs the bootstrap for every row in df and appends the summary columns.
    term_col holds strings like "Term A | Term B" (pairs) or
    "Term A | Term B | Term C" (triplets), which are split back into
    individual term names. Returns df unchanged if it is empty or does not
    contain term_col (e.g. no pairs/triplets met the minimum threshold).
    """
    if df.empty or term_col not in df.columns:
        return df

    rows_out = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        term_names = [t.strip() for t in row[term_col].split("|")]
        assert len(term_names) == n_terms, f"Expected {n_terms} terms, got {term_names}"

        pmi_diffs = bootstrap_pmi_difference(subgroup_terms, control_terms, term_names, n_boot, seed)
        rows_out.append(summarize_bootstrap(pmi_diffs))

        if i % 10 == 0 or i == total:
            print(f"  bootstrap progress: {i}/{total}")

    boot_df = pd.DataFrame(rows_out, index=df.index)
    return pd.concat([df, boot_df], axis=1)


# ---------------------------------------------------------------------------
# Formatting for submission
# ---------------------------------------------------------------------------

def format_for_supplementary(df, term_col):
    """Renames and reorders columns into a clean, submission-ready table.
    Returns an empty table with the expected column headers if no
    pairs/triplets were found (e.g. SNOMED-only triplet analysis)."""

    expected_columns = [
        "Finding", "N patients AL (with finding)", "N patients AL (total tested)", "% AL",
        "N controls (with finding)", "N controls (total tested)", "% controls",
        "PMI difference (AL - controls)", "FDR-corrected p",
        "Bootstrap PMI difference (median, 95% CI)", "% bootstrap iterations PMI diff > 0",
        "% bootstrap iterations undefined",
    ]

    if df.empty or term_col not in df.columns:
        return pd.DataFrame(columns=expected_columns)

    out = pd.DataFrame()
    out["Finding"] = df[term_col]
    out["N patients AL (with finding)"] = df["subgroup_count"]
    out["N patients AL (total tested)"] = df["subgroup_total_patients"]
    out["% AL"] = df["pct_subgroup"].round(1)
    out["N controls (with finding)"] = df["control_count"]
    out["N controls (total tested)"] = df["control_total_patients"]
    out["% controls"] = df["pct_control"].round(1)
    out["PMI difference (AL - controls)"] = df["pmi_difference"].round(2)
    out["FDR-corrected p"] = df["p_value_corrected"].apply(
        lambda p: "<0.001" if pd.notna(p) and p < 0.001 else (round(p, 3) if pd.notna(p) else np.nan)
    )
    out["Bootstrap PMI difference (median, 95% CI)"] = df.apply(
        lambda r: f"{r['boot_pmi_diff_median']:.2f} ({r['boot_pmi_diff_ci_low']:.2f}-{r['boot_pmi_diff_ci_high']:.2f})"
        if pd.notna(r["boot_pmi_diff_median"]) else "undefined",
        axis=1,
    )
    out["% bootstrap iterations PMI diff > 0"] = df["boot_pct_pmi_diff_positive"].round(1)
    out["% bootstrap iterations undefined"] = df["boot_pct_undefined"].round(1)

    out = out.sort_values("FDR-corrected p", key=lambda col: col.apply(
        lambda v: -1 if v == "<0.001" else v
    ))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_control_terms(control_df, min_terms):
    terms_by_patient = {}
    for pid, visits in control_df.groupby("Patient_ID"):
        terms = set()
        for _, visit in visits.iterrows():
            if pd.notna(visit["Diagnoses"]):
                terms.update(t.strip() for t in str(visit["Diagnoses"]).split("|") if t.strip())
        if len(terms) >= min_terms:
            terms_by_patient[pid] = terms
    return terms_by_patient


def run_full_analysis(analysis_name, include_lab_terms):
    """Runs the effect-size and bootstrap pipeline for one analysis mode
    (SNOMED-only or lab-augmented), mirroring the original pipeline's
    two-pass structure."""

    print(f"\n{'=' * 60}")
    print(f"Analysis mode: {analysis_name}")
    print(f"{'=' * 60}")

    patient_subgroups = load_and_classify_patients()
    amyloid_df, control_df = load_and_augment_data(include_lab_terms=include_lab_terms)

    subgroup_ids = patient_subgroups[SUBGROUP]
    subgroup_terms_2 = aggregate_patient_terms(amyloid_df, subgroup_ids, SUBGROUP, min_terms=2)
    subgroup_terms_3 = {pid: t for pid, t in subgroup_terms_2.items() if len(t) >= 3}

    control_terms_2 = build_control_terms(control_df, min_terms=2)
    control_terms_3 = {pid: t for pid, t in control_terms_2.items() if len(t) >= 3}

    print(f"AL patients: {len(subgroup_terms_2)} (>=2 terms), {len(subgroup_terms_3)} (>=3 terms)")
    print(f"Controls: {len(control_terms_2)} (>=2 terms), {len(control_terms_3)} (>=3 terms)")

    sub_2pmi = calculate_2term_pmi(subgroup_terms_2, SUBGROUP)
    ctrl_2pmi = calculate_2term_pmi(control_terms_2, "Controls")
    comparison_2pmi = compare_2pmi(sub_2pmi, ctrl_2pmi, SUBGROUP, min_patients=MIN_PATIENTS_THRESHOLD)

    sub_3pmi = calculate_3term_pmi(subgroup_terms_3, SUBGROUP)
    ctrl_3pmi = calculate_3term_pmi(control_terms_3, "Controls")
    comparison_3pmi = compare_3pmi(sub_3pmi, ctrl_3pmi, SUBGROUP, min_patients=MIN_PATIENTS_THRESHOLD)

    print(f"Pairs to process: {len(comparison_2pmi)}")
    print(f"Triplets to process: {len(comparison_3pmi)}")
    print(f"Bootstrap iterations per finding: {N_BOOTSTRAP}\n")

    enriched_2pmi = add_percentages(
        comparison_2pmi, "subgroup_count", "subgroup_total_patients",
        "control_count", "control_total_patients"
    )
    enriched_3pmi = add_percentages(
        comparison_3pmi, "subgroup_count", "subgroup_total_patients",
        "control_count", "control_total_patients"
    )

    print("Running bootstrap for all pairs...")
    enriched_2pmi = add_bootstrap_columns(
        enriched_2pmi, "term_pair", subgroup_terms_2, control_terms_2,
        n_terms=2, n_boot=N_BOOTSTRAP, seed=SEED
    )

    print("Running bootstrap for all triplets...")
    enriched_3pmi = add_bootstrap_columns(
        enriched_3pmi, "term_triplet", subgroup_terms_3, control_terms_3,
        n_terms=3, n_boot=N_BOOTSTRAP, seed=SEED
    )

    pairs_final = format_for_supplementary(enriched_2pmi, "term_pair")
    triplets_final = format_for_supplementary(enriched_3pmi, "term_triplet")

    return pairs_final, triplets_final


def main():
    pairs_snomed_only, triplets_snomed_only = run_full_analysis("SNOMED_ONLY", include_lab_terms=False)
    pairs_with_labs, triplets_with_labs = run_full_analysis("WITH_LABS", include_lab_terms=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{OUTPUT_DIR}PMI_Supplementary_Evidence_{timestamp}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pairs_snomed_only.to_excel(writer, sheet_name="Pairs_SNOMED_only", index=False)
        triplets_snomed_only.to_excel(writer, sheet_name="Triplets_SNOMED_only", index=False)
        pairs_with_labs.to_excel(writer, sheet_name="Pairs_SNOMED_plus_labs", index=False)
        triplets_with_labs.to_excel(writer, sheet_name="Triplets_SNOMED_plus_labs", index=False)

        notes = pd.DataFrame({
            "Note": [
                f"Minimum patient threshold: a pair/triplet was tested only if present in "
                f">= {MIN_PATIENTS_THRESHOLD} patients in both the {SUBGROUP} group and the "
                f"control group.",
                "PMI difference (AL - controls) is the point estimate computed directly "
                "from the observed data, as in the original PMI pipeline.",
                f"Bootstrap: {N_BOOTSTRAP} resamples with replacement, drawn independently from "
                f"the {SUBGROUP} and control cohorts (group membership preserved), recomputing "
                "the PMI difference between groups at each iteration. Reported as the median "
                "and 95% interval (2.5th-97.5th percentile) across valid iterations.",
                "'% bootstrap iterations undefined' reflects resamples where zero co-occurrence "
                "of the finding occurred in one or both groups, making PMI undefined for that "
                "iteration; a high value indicates a sparse underlying finding.",
                "SNOMED_only sheets correspond to Figure 4A (coded diagnoses only); "
                "SNOMED_plus_labs sheets correspond to Figure 4B/4C (augmented with "
                "lab-derived phenotypes for proteinuria, nephrotic-range proteinuria, and "
                "microscopic haematuria).",
            ]
        })
        notes.to_excel(writer, sheet_name="Methods_Notes", index=False)

    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
