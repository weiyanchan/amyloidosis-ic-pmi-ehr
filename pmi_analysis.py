"""
Core PMI pipeline: 2-term and 3-term pointwise mutual information comparing
each amyloidosis subgroup against pre-reference-matched controls, restricted
to pre-diagnosis/pre-reference visits.

Two modes are run (see main()):
  - SNOMED_ONLY: coded SNOMED-CT diagnoses only (Figure 4A)
  - WITH_LABS: SNOMED-CT augmented with lab-derived pseudo-terms for
    proteinuria, nephrotic-range proteinuria, and microscopic haematuria
    (Figure 4B/4C)

Lab-derived term definitions (Methods):
  - Proteinuria: urine protein concentration > 0.15 g/day
  - Nephrotic-range proteinuria: urine protein concentration > 3.5 g/day
  - Microscopic haematuria: urine red blood cell count >= 3 RBC/HPF

This module is also imported by pmi_effect_estimates.py for the
effect-size/bootstrap analysis (Supplementary Tables).
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

from config import IC_EXCEL_FILE, CLASSIFICATION_FILE, LAB_FILE, OUTPUT_DIR, \
    PROTEINURIA_THRESHOLD, NEPHROTIC_THRESHOLD, MICROSCOPIC_HAEMATURIA_THRESHOLD
from utils import parse_pipe_separated

MICROSCOPIC_HAEMATURIA_BASE = "Microscopic haematuria"
CHEST_PAIN_CODES = ["Chest pain", "Atypical chest pain"]

ANALYSIS_LABELS = {"WITH_LABS": "SNOMED-CT + Labs", "SNOMED_ONLY": "SNOMED-CT Only"}

SUBGROUP_NAMES = ['AL_Amyloidosis', 'ATTRv_Amyloidosis', 'ATTRwt_Amyloidosis',
                  'CAA', 'Gelsolin_Amyloidosis', 'ATTR_Amyloidosis_Combined']


# ---------------------------------------------------------------------------
# Patient classification
# ---------------------------------------------------------------------------

def load_and_classify_patients():
    df = pd.read_excel(CLASSIFICATION_FILE, sheet_name='NEW_Amyloid_Patients')
    print(f"Loaded {len(df):,} patients")

    patient_subgroups = {name: [] for name in SUBGROUP_NAMES}

    for _, row in df.iterrows():
        patient_id = str(row['Patient_ID'])
        subtype = row['Classified_Subtype']

        if subtype in ['AL amyloidosis', 'ATTRwt-Likely AL', 'Likely AL']:
            patient_subgroups['AL_Amyloidosis'].append(patient_id)
        elif subtype == 'ATTRv':
            patient_subgroups['ATTRv_Amyloidosis'].append(patient_id)
            patient_subgroups['ATTR_Amyloidosis_Combined'].append(patient_id)
        elif subtype == 'ATTRwt':
            patient_subgroups['ATTRwt_Amyloidosis'].append(patient_id)
            patient_subgroups['ATTR_Amyloidosis_Combined'].append(patient_id)
        elif subtype == 'ATTR':
            patient_subgroups['ATTR_Amyloidosis_Combined'].append(patient_id)
        elif subtype == 'CAA':
            patient_subgroups['CAA'].append(patient_id)
        elif subtype == 'Gelsolin':
            patient_subgroups['Gelsolin_Amyloidosis'].append(patient_id)

    for subgroup, patients in patient_subgroups.items():
        print(f"  {subgroup.replace('_', ' ')}: {len(patients)}")

    return patient_subgroups


# ---------------------------------------------------------------------------
# Lab-to-SNOMED augmentation
# ---------------------------------------------------------------------------

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


def convert_labs_to_snomed_terms(lab_df):
    """Maps lab results to pseudo-SNOMED terms per the thresholds defined
    in Methods: proteinuria >0.15 g/day, nephrotic-range proteinuria
    >3.5 g/day, microscopic haematuria >=3 RBC/HPF."""

    lab_snomed_mapping = {}

    for _, row in lab_df.iterrows():
        visit_id = row['Visit_ID']
        tpu = parse_lab_value(row.get('TPU', np.nan))
        pu = parse_lab_value(row.get('PU', np.nan))
        p24u = parse_lab_value(row.get('P24U', np.nan))
        urbc = parse_lab_value(row.get('URBC', np.nan))

        lab_terms = []

        proteinuria_value = tpu if not np.isnan(tpu) else pu
        if not np.isnan(proteinuria_value):
            if proteinuria_value > NEPHROTIC_THRESHOLD:
                lab_terms.append('Nephrotic range proteinuria')
            elif proteinuria_value > PROTEINURIA_THRESHOLD:
                lab_terms.append('Proteinuria')

        if not np.isnan(p24u):
            if p24u > NEPHROTIC_THRESHOLD and 'Nephrotic range proteinuria' not in lab_terms:
                lab_terms.append('Nephrotic range proteinuria')
            elif p24u > PROTEINURIA_THRESHOLD and not lab_terms:
                lab_terms.append('Proteinuria')

        if not np.isnan(urbc) and urbc >= MICROSCOPIC_HAEMATURIA_THRESHOLD:
            lab_terms.append('Microscopic haematuria')

        if lab_terms:
            lab_snomed_mapping[visit_id] = lab_terms

    n_proteinuria = sum('Proteinuria' in t for t in lab_snomed_mapping.values())
    n_nephrotic = sum('Nephrotic range proteinuria' in t for t in lab_snomed_mapping.values())
    n_haematuria = sum('Microscopic haematuria' in t for t in lab_snomed_mapping.values())

    print(f"Converted {len(lab_snomed_mapping)} visits to lab-derived SNOMED terms "
          f"(proteinuria={n_proteinuria}, nephrotic={n_nephrotic}, "
          f"microscopic haematuria={n_haematuria})")

    return lab_snomed_mapping


def augment_snomed_with_labs(diagnosis_df, lab_snomed_mapping):
    augmented_df = diagnosis_df.copy()
    n_augmented, n_new_terms = 0, 0

    for idx, row in augmented_df.iterrows():
        lab_terms = lab_snomed_mapping.get(row['Visit_ID'])
        if not lab_terms:
            continue

        existing_terms = set(parse_pipe_separated(row['SNOMED_Descriptions']))
        terms_to_add = [t for t in lab_terms
                         if not any(t.lower() in e.lower() for e in existing_terms)]

        if terms_to_add:
            augmented_df.at[idx, 'SNOMED_Descriptions'] = ' | '.join(list(existing_terms) + terms_to_add)
            n_augmented += 1
            n_new_terms += len(terms_to_add)

    print(f"Augmented {n_augmented} visits with {n_new_terms} new lab-derived terms")
    return augmented_df


def consolidate_duplicate_term(diagnosis_df, term):
    """Collapses multiple occurrences of the same term (e.g. one from
    SNOMED coding and one from lab augmentation) into a single instance."""

    consolidated_df = diagnosis_df.copy()
    n_consolidated = 0

    for idx, row in consolidated_df.iterrows():
        terms = parse_pipe_separated(row['SNOMED_Descriptions'])
        if terms.count(term) > 1:
            other_terms = [t for t in terms if t != term]
            consolidated_df.at[idx, 'SNOMED_Descriptions'] = ' | '.join(other_terms + [term])
            n_consolidated += 1

    print(f"Consolidated '{term}' in {n_consolidated} visits")
    return consolidated_df


def consolidate_chest_pain(diagnosis_df):
    """Merges 'Chest pain' and 'Atypical chest pain' into a single 'Chest
    pain' term."""

    consolidated_df = diagnosis_df.copy()
    n_consolidated = 0

    for idx, row in consolidated_df.iterrows():
        terms = parse_pipe_separated(row['SNOMED_Descriptions'])
        has_chest_pain = any(t in CHEST_PAIN_CODES for t in terms)
        if not has_chest_pain:
            continue

        other_terms = [t for t in terms if t not in CHEST_PAIN_CODES]
        if 'Chest pain' not in other_terms:
            other_terms.append('Chest pain')
        consolidated_df.at[idx, 'SNOMED_Descriptions'] = ' | '.join(other_terms)
        n_consolidated += 1

    print(f"Consolidated chest pain variants in {n_consolidated} visits")
    return consolidated_df


def load_and_augment_data(include_lab_terms=True):
    """Loads pre-diagnosis/pre-reference visits, optionally augmenting with
    lab-derived terms and consolidating duplicate/variant terms."""

    amyloid_df = pd.read_excel(IC_EXCEL_FILE, sheet_name='Amyloid_449_Visit_IC_ENTIRE')
    control_df = pd.read_excel(IC_EXCEL_FILE, sheet_name='Control_NEW_VARIANCE_Visit')

    amyloid_pre = amyloid_df[amyloid_df['Months_From_Diagnosis'] < 0].copy()
    control_pre = control_df[control_df['Months_From_Reference'] < 0].copy()
    print(f"Amyloidosis (pre-diagnosis): {len(amyloid_pre)} visits; "
          f"Controls (pre-reference): {len(control_pre)} visits")

    if include_lab_terms:
        al_labs = pd.read_excel(LAB_FILE, sheet_name='AL_Amyloidosis')
        attr_labs = pd.read_excel(LAB_FILE, sheet_name='ATTR_Amyloidosis')
        control_labs = pd.read_excel(LAB_FILE, sheet_name='Controls')
        all_labs = pd.concat([al_labs, attr_labs, control_labs], ignore_index=True)

        lab_snomed_mapping = convert_labs_to_snomed_terms(all_labs)

        amyloid_aug = augment_snomed_with_labs(amyloid_pre, lab_snomed_mapping)
        control_aug = augment_snomed_with_labs(control_pre, lab_snomed_mapping)

        amyloid_aug = consolidate_duplicate_term(amyloid_aug, MICROSCOPIC_HAEMATURIA_BASE)
        control_aug = consolidate_duplicate_term(control_aug, MICROSCOPIC_HAEMATURIA_BASE)
    else:
        amyloid_aug = amyloid_pre.copy()
        control_aug = control_pre.copy()

    amyloid_aug = consolidate_chest_pain(amyloid_aug)
    control_aug = consolidate_chest_pain(control_aug)

    for df in (amyloid_aug, control_aug):
        df['Patient_ID'] = df['Patient_ID'].astype(str)
        df['Diagnoses'] = df['SNOMED_Descriptions']

    print(f"Final: Amyloidosis {len(amyloid_aug)} visits ({amyloid_aug['Patient_ID'].nunique()} patients), "
          f"Controls {len(control_aug)} visits ({control_aug['Patient_ID'].nunique()} patients)")

    return amyloid_aug, control_aug


# ---------------------------------------------------------------------------
# PMI calculation
# ---------------------------------------------------------------------------

def aggregate_patient_terms(df, patient_ids, subgroup_name, min_terms=2):
    subgroup_df = df[df['Patient_ID'].isin(patient_ids)]
    if subgroup_df.empty:
        return {}

    patient_terms = {}
    for patient_id, patient_visits in subgroup_df.groupby('Patient_ID'):
        all_terms = set()
        for _, visit in patient_visits.iterrows():
            all_terms.update(parse_pipe_separated(visit['Diagnoses']))
        if len(all_terms) >= min_terms:
            patient_terms[patient_id] = all_terms

    return patient_terms


def calculate_2term_pmi(patient_terms_dict, group_name):
    # group_name is unused here but kept for interface consistency with
    # downstream scripts (pmi_effect_estimates.py) that call this function.
    if not patient_terms_dict:
        return {}

    total_patients = len(patient_terms_dict)
    term_counts = {}
    for terms in patient_terms_dict.values():
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

    pair_counts = {}
    for terms in patient_terms_dict.values():
        if len(terms) < 2:
            continue
        for term1, term2 in combinations(terms, 2):
            pair = tuple(sorted([term1, term2]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    pmi_results = {}
    for pair, pair_count in pair_counts.items():
        term1, term2 = pair
        p1 = term_counts[term1] / total_patients
        p2 = term_counts[term2] / total_patients
        p_both = pair_count / total_patients

        pmi_results[pair] = {
            'pmi_2term': np.log2(p_both / (p1 * p2)),
            'pair_count': pair_count,
            'term1_count': term_counts[term1],
            'term2_count': term_counts[term2],
            'total_patients': total_patients,
        }

    return pmi_results


def calculate_3term_pmi(patient_terms_dict, group_name):
    if not patient_terms_dict:
        return {}

    total_patients = len(patient_terms_dict)
    term_counts = {}
    for terms in patient_terms_dict.values():
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

    triplet_counts = {}
    for terms in patient_terms_dict.values():
        if len(terms) < 3:
            continue
        for triplet in combinations(terms, 3):
            key = tuple(sorted(triplet))
            triplet_counts[key] = triplet_counts.get(key, 0) + 1

    pmi_results = {}
    for triplet, triplet_count in triplet_counts.items():
        t1, t2, t3 = triplet
        p1, p2, p3 = (term_counts[t] / total_patients for t in triplet)
        p_all = triplet_count / total_patients

        pmi_results[triplet] = {
            'pmi_3term': np.log2(p_all / (p1 * p2 * p3)),
            'triplet_count': triplet_count,
            'total_patients': total_patients,
        }

    return pmi_results


def _contingency_test(subgroup_count, subgroup_total, control_count, control_total):
    contingency = np.array([
        [subgroup_count, subgroup_total - subgroup_count],
        [control_count, control_total - control_count],
    ])

    if np.any(contingency < 0):
        return np.nan, np.nan

    a, b = contingency[0]
    c, d = contingency[1]
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else np.inf

    if contingency.min() < 5:
        try:
            _, p_value = fisher_exact(contingency)
        except ValueError:
            return np.nan, np.nan
    else:
        try:
            _, p_value, _, _ = chi2_contingency(contingency)
        except ValueError:
            return np.nan, np.nan

    return p_value, odds_ratio


def compare_2pmi(subgroup_pmi, control_pmi, subgroup_name, min_patients=2):
    common_pairs = [
        p for p in set(subgroup_pmi) & set(control_pmi)
        if subgroup_pmi[p]['pair_count'] >= min_patients
        and control_pmi[p]['pair_count'] >= min_patients
    ]
    if not common_pairs:
        return pd.DataFrame()

    results = []
    for pair in common_pairs:
        p_value, odds_ratio = _contingency_test(
            subgroup_pmi[pair]['pair_count'], subgroup_pmi[pair]['total_patients'],
            control_pmi[pair]['pair_count'], control_pmi[pair]['total_patients'],
        )
        results.append({
            'subgroup': subgroup_name,
            'term_pair': f"{pair[0]} | {pair[1]}",
            'subgroup_2pmi': subgroup_pmi[pair]['pmi_2term'],
            'control_2pmi': control_pmi[pair]['pmi_2term'],
            'pmi_difference': subgroup_pmi[pair]['pmi_2term'] - control_pmi[pair]['pmi_2term'],
            'subgroup_count': subgroup_pmi[pair]['pair_count'],
            'control_count': control_pmi[pair]['pair_count'],
            'subgroup_total_patients': subgroup_pmi[pair]['total_patients'],
            'control_total_patients': control_pmi[pair]['total_patients'],
            'p_value': p_value,
            'odds_ratio': odds_ratio,
        })

    return _add_fdr_correction(pd.DataFrame(results))


def compare_3pmi(subgroup_pmi, control_pmi, subgroup_name, min_patients=2):
    common_triplets = [
        t for t in set(subgroup_pmi) & set(control_pmi)
        if subgroup_pmi[t]['triplet_count'] >= min_patients
        and control_pmi[t]['triplet_count'] >= min_patients
    ]
    if not common_triplets:
        return pd.DataFrame()

    results = []
    for triplet in common_triplets:
        p_value, odds_ratio = _contingency_test(
            subgroup_pmi[triplet]['triplet_count'], subgroup_pmi[triplet]['total_patients'],
            control_pmi[triplet]['triplet_count'], control_pmi[triplet]['total_patients'],
        )
        results.append({
            'subgroup': subgroup_name,
            'term_triplet': f"{triplet[0]} | {triplet[1]} | {triplet[2]}",
            'subgroup_3pmi': subgroup_pmi[triplet]['pmi_3term'],
            'control_3pmi': control_pmi[triplet]['pmi_3term'],
            'pmi_difference': subgroup_pmi[triplet]['pmi_3term'] - control_pmi[triplet]['pmi_3term'],
            'subgroup_count': subgroup_pmi[triplet]['triplet_count'],
            'control_count': control_pmi[triplet]['triplet_count'],
            'subgroup_total_patients': subgroup_pmi[triplet]['total_patients'],
            'control_total_patients': control_pmi[triplet]['total_patients'],
            'p_value': p_value,
            'odds_ratio': odds_ratio,
        })

    return _add_fdr_correction(pd.DataFrame(results))


def _add_fdr_correction(results_df):
    valid_p = results_df['p_value'].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, method='fdr_bh')[1]
        results_df.loc[valid_p.index, 'p_value_corrected'] = corrected
    return results_df


# ---------------------------------------------------------------------------
# Volcano plots
# ---------------------------------------------------------------------------

def create_pmi_volcano_plots(all_results, timestamp, analysis_suffix, n_terms):
    """Creates one volcano plot per subgroup for either 2-term or 3-term
    PMI results (n_terms=2 or 3)."""

    label = ANALYSIS_LABELS.get(analysis_suffix, analysis_suffix)
    color_thresholds = [(0.001, 'red'), (0.01, 'orange'), (0.05, 'gold')]

    for subgroup in SUBGROUP_NAMES:
        fig, ax = plt.subplots(figsize=(14, 10))
        title = f'Pre-diagnosis {subgroup.replace("_", " ")} vs Pre-reference Controls ({n_terms}PMI {label})'
        filename = os.path.join(
            OUTPUT_DIR, f'{subgroup}_vs_Controls_{analysis_suffix}_{n_terms}PMI_{timestamp}.png'
        )

        results_df = all_results.get(subgroup, pd.DataFrame())
        plot_data = results_df.dropna(subset=['p_value_corrected']) if not results_df.empty else pd.DataFrame()

        if plot_data.empty:
            message = 'No data available' if results_df.empty else 'No statistically testable results'
            ax.text(0.5, 0.5, f'{message}\nfor {subgroup.replace("_", " ")}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=16,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            plot_data = plot_data.copy()
            plot_data['-log10_p'] = -np.log10(plot_data['p_value_corrected'])

            def color_for(p):
                return next((c for t, c in color_thresholds if p < t), 'lightgray')

            colors = plot_data['p_value_corrected'].apply(color_for)

            ax.scatter(plot_data['pmi_difference'], plot_data['-log10_p'],
                       c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.8)
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.8, linewidth=2,
                       label='p = 0.05 threshold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.4, linewidth=1)

            legend_elements = [
                plt.scatter([], [], c=c, s=100, label=lbl) for c, lbl in
                [('red', 'p < 0.001 (***)'), ('orange', 'p < 0.01 (**)'),
                 ('gold', 'p < 0.05 (*)'), ('lightgray', 'p >= 0.05 (ns)')]
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

            ax.set_xlabel(f'{n_terms}PMI Difference (Subgroup - Control)', fontsize=14, fontweight='bold')
            ax.set_ylabel('-log10(Corrected P-value)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"{n_terms}PMI volcano plots saved for {len(SUBGROUP_NAMES)} subgroups")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pmi_analysis(analysis_name, include_lab_terms=True):
    print(f"\nPMI analysis: {analysis_name}")
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    patient_subgroups = load_and_classify_patients()
    amyloid_df, control_df = load_and_augment_data(include_lab_terms=include_lab_terms)

    control_terms_2 = {}
    for pid, visits in control_df.groupby('Patient_ID'):
        terms = set()
        for _, visit in visits.iterrows():
            terms.update(parse_pipe_separated(visit['Diagnoses']))
        if len(terms) >= 2:
            control_terms_2[pid] = terms

    control_2pmi = calculate_2term_pmi(control_terms_2, "Controls")
    control_terms_3 = {pid: t for pid, t in control_terms_2.items() if len(t) >= 3}
    control_3pmi = calculate_3term_pmi(control_terms_3, "Controls")
    print(f"Control 2PMI: {len(control_2pmi)} pairs, 3PMI: {len(control_3pmi)} triplets")

    all_2pmi_results, all_3pmi_results = {}, {}

    for subgroup_name, patient_ids in patient_subgroups.items():
        if not patient_ids:
            continue

        terms_2 = aggregate_patient_terms(amyloid_df, patient_ids, subgroup_name, min_terms=2)
        if terms_2:
            comparison = compare_2pmi(calculate_2term_pmi(terms_2, subgroup_name), control_2pmi, subgroup_name)
            if not comparison.empty:
                all_2pmi_results[subgroup_name] = comparison

        terms_3 = aggregate_patient_terms(amyloid_df, patient_ids, subgroup_name, min_terms=3)
        if terms_3:
            comparison = compare_3pmi(calculate_3term_pmi(terms_3, subgroup_name), control_3pmi, subgroup_name)
            if not comparison.empty:
                all_3pmi_results[subgroup_name] = comparison

    print(f"Subgroups with pairs: {len(all_2pmi_results)}, with triplets: {len(all_3pmi_results)}")

    output_2pmi = os.path.join(OUTPUT_DIR, f"PMI_{analysis_name}_2pmi_{timestamp}.xlsx")
    output_3pmi = os.path.join(OUTPUT_DIR, f"PMI_{analysis_name}_3pmi_{timestamp}.xlsx")

    for output_file, results, count_label in [
        (output_2pmi, all_2pmi_results, 'Pairs_Analyzed'),
        (output_3pmi, all_3pmi_results, 'Triplets_Analyzed'),
    ]:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary = []
            for sg, res in results.items():
                summary.append({
                    'Subgroup': sg, count_label: len(res),
                    'Significant_p<0.05': (res['p_value_corrected'] < 0.05).sum()
                    if 'p_value_corrected' in res.columns else 0,
                })
                res.sort_values('pmi_difference', ascending=False).to_excel(
                    writer, sheet_name=sg[:31], index=False
                )
            if summary:
                pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)

    print(f"Results saved: {output_2pmi}, {output_3pmi}")

    create_pmi_volcano_plots(all_2pmi_results, timestamp, analysis_name, n_terms=2)
    create_pmi_volcano_plots(all_3pmi_results, timestamp, analysis_name, n_terms=3)

    return all_2pmi_results, all_3pmi_results


def main():
    run_pmi_analysis("WITH_LABS", include_lab_terms=True)
    run_pmi_analysis("SNOMED_ONLY", include_lab_terms=False)


if __name__ == "__main__":
    main()
