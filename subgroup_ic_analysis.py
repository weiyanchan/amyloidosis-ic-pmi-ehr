"""
Subgroup IC comparison: Controls vs. Localised Cutaneous vs. AL Amyloidosis
(incl. Likely AL) vs. ATTR Amyloidosis vs. Other Amyloidosis.

NOTE: this script depends on a cache entry "diagnosis_ic_NEW_variance_matched"
containing visit-level IC data, keyed under the same cache_key used by
matching.py. That cache entry is produced by a separate IC-computation
script not yet included in this repository -- add it once available.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, gaussian_kde
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

from config import CACHE_DIR, CLASSIFICATION_FILE, OUTPUT_DIR, PATIENT_DATA_FILE, LAB_DATA_FILES
from matching import get_cache_key
from utils import (
    load_from_cache, parse_ic_values, parse_pipe_separated,
    filter_out_amyloid_terms, get_significance_asterisks,
)

GROUP_KEYS = ['controls', 'localised_cutaneous', 'al_amyloidosis_incl_likely',
              'attr_amyloidosis', 'other_amyloidosis']
GROUP_NAMES = ['Controls', 'Localised Cutaneous', 'AL Amyloidosis (incl Likely AL)',
               'ATTR Amyloidosis', 'Other Amyloidosis']

COLORS = dict(zip(GROUP_NAMES, ['#4472C4', '#54a1a1', '#ea801c', '#9B59B6', '#E74C3C']))

SUBTYPE_MAPPING = {
    'Skin': 'Localised Cutaneous',
    'AL amyloidosis': 'AL Amyloidosis (incl Likely AL)',
    'ATTRwt-Likely AL': 'AL Amyloidosis (incl Likely AL)',
    'Likely AL': 'AL Amyloidosis (incl Likely AL)',
    'ATTRwt': 'ATTR Amyloidosis',
    'ATTRv': 'ATTR Amyloidosis',
    'ATTR': 'ATTR Amyloidosis',
    'Others': 'Other Amyloidosis',
    'CAA': 'Other Amyloidosis',
    'Gelsolin': 'Other Amyloidosis',
}

GROUP_KEY_BY_NAME = {
    'Localised Cutaneous': 'localised_cutaneous',
    'AL Amyloidosis (incl Likely AL)': 'al_amyloidosis_incl_likely',
    'ATTR Amyloidosis': 'attr_amyloidosis',
    'Other Amyloidosis': 'other_amyloidosis',
}


def load_patient_classifications():
    classification_df = pd.read_excel(CLASSIFICATION_FILE, sheet_name='NEW_Amyloid_Patients')

    patient_groups = {key: [] for key in GROUP_KEY_BY_NAME.values()}

    for _, row in classification_df.iterrows():
        subtype = row['Classified_Subtype']
        group_name = SUBTYPE_MAPPING.get(subtype)
        if group_name is None:
            print(f"Unknown subtype '{subtype}' for patient {row['Patient_ID']}")
            continue
        patient_groups[GROUP_KEY_BY_NAME[group_name]].append(str(row['Patient_ID']))

    for name, key in GROUP_KEY_BY_NAME.items():
        print(f"{name}: {len(patient_groups[key])} patients")

    return patient_groups


def load_visit_level_ic_data(cache_key):
    cached = load_from_cache(CACHE_DIR, cache_key, "diagnosis_ic_NEW_variance_matched")
    if cached is None:
        raise FileNotFoundError(
            "Cache entry 'diagnosis_ic_NEW_variance_matched' not found. "
            "This must be produced by the visit-level IC computation script."
        )

    amyl_df = pd.DataFrame(cached['amyloid_visit_data'])
    control_df = pd.DataFrame(cached['control_visit_data'])

    if 'Months_From_Diagnosis' in amyl_df.columns:
        amyl_df['Months_From_Reference'] = amyl_df['Months_From_Diagnosis']

    control_ids = {str(v['Patient_ID']) for v in cached['control_visit_data'] if 'Patient_ID' in v}

    print(f"Amyloidosis: {len(amyl_df):,} visits")
    print(f"Matched controls: {len(control_df):,} visits, {len(control_ids):,} patients")

    return amyl_df, control_df, control_ids


def calculate_patient_ic(amyl_visits, control_visits):
    """Patient-level mean IC restricted to pre-diagnosis/pre-reference visits.
    Amyloid-related SNOMED terms are filtered from both groups."""

    rows = []

    for label, visits in [('Amyloidosis', amyl_visits), ('Control', control_visits)]:
        pre = visits[visits['Months_From_Reference'] < 0].copy()
        n_filtered, n_total = 0, 0

        for patient_id, patient_visits in pre.groupby('Patient_ID'):
            patient_ic_values = []

            for _, visit in patient_visits.iterrows():
                if pd.isna(visit['IC_Values']):
                    continue
                ic_values = parse_ic_values(visit['IC_Values'])

                if pd.notna(visit.get('SNOMED_Descriptions')):
                    descriptions = parse_pipe_separated(visit['SNOMED_Descriptions'])
                    n_total += len(ic_values)
                    ic_values, filtered = filter_out_amyloid_terms(ic_values, descriptions)
                    n_filtered += filtered

                patient_ic_values.extend(ic_values)

            if patient_ic_values:
                rows.append({
                    'Patient_ID': str(patient_id),
                    'Patient_Type': label,
                    'Total_Visits': len(patient_visits),
                    'Total_SNOMED_Terms': len(patient_ic_values),
                    'Mean_IC_Per_Patient': np.mean(patient_ic_values),
                })

        print(f"{label}: {sum(1 for r in rows if r['Patient_Type'] == label):,} patients with valid IC"
              + (f" ({n_filtered:,}/{n_total:,} amyloid terms filtered)" if n_total else ""))

    return pd.DataFrame(rows)


def build_subgroup_data(patient_groups, patient_ic_df, control_ids):
    patient_ic_df = patient_ic_df.copy()
    patient_ic_df['Patient_ID'] = patient_ic_df['Patient_ID'].astype(str)

    amyl_df = patient_ic_df[patient_ic_df['Patient_Type'] == 'Amyloidosis']
    control_df = patient_ic_df[
        (patient_ic_df['Patient_Type'] == 'Control') & (patient_ic_df['Patient_ID'].isin(control_ids))
    ].copy()
    control_df['Subgroup'] = 'Controls'

    subgroup_data = {'controls': control_df}
    for key, patient_ids in patient_groups.items():
        subset = amyl_df[amyl_df['Patient_ID'].isin(set(patient_ids))].copy()
        group_name = next(n for n, k in GROUP_KEY_BY_NAME.items() if k == key)
        subset['Subgroup'] = group_name
        subgroup_data[key] = subset
        print(f"{group_name}: n={len(subset)}")

    print(f"Controls: n={len(control_df):,}")
    return subgroup_data


def get_group_values(subgroup_data, key):
    if key in subgroup_data and len(subgroup_data[key]) > 0:
        return subgroup_data[key]['Mean_IC_Per_Patient'].dropna()
    return pd.Series([], dtype=float)


def run_statistical_analysis(subgroup_data):
    group_arrays = [get_group_values(subgroup_data, k) for k in GROUP_KEYS]

    descriptive = []
    for name, data in zip(GROUP_NAMES, group_arrays):
        if len(data) > 0:
            descriptive.append({
                'Group': name, 'N': len(data), 'Mean': round(data.mean(), 6),
                'SD': round(data.std(ddof=1), 6), 'Median': round(data.median(), 6),
                'Q1': round(data.quantile(0.25), 6), 'Q3': round(data.quantile(0.75), 6),
                'Min': round(data.min(), 6), 'Max': round(data.max(), 6),
            })
        else:
            descriptive.append({'Group': name, 'N': 0, 'Mean': np.nan, 'SD': np.nan,
                                 'Median': np.nan, 'Q1': np.nan, 'Q3': np.nan, 'Min': np.nan, 'Max': np.nan})

    descriptive_df = pd.DataFrame(descriptive)
    valid = [d for d in group_arrays if len(d) > 0]

    if len(valid) < 2:
        print("Insufficient groups for statistical testing")
        return descriptive_df, pd.DataFrame(), None, None

    kw_stat, kw_p = kruskal(*valid)
    print(f"Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.8f} "
          f"({'significant' if kw_p < 0.05 else 'not significant'})")

    if kw_p >= 0.05:
        return descriptive_df, pd.DataFrame(), kw_stat, kw_p

    pairwise, raw_p_values = [], []
    for i in range(len(GROUP_KEYS)):
        for j in range(i + 1, len(GROUP_KEYS)):
            d1, d2 = group_arrays[i], group_arrays[j]
            if len(d1) == 0 or len(d2) == 0:
                continue

            u_stat, raw_p = mannwhitneyu(d1, d2, alternative='two-sided')
            n1, n2 = len(d1), len(d2)
            z = abs(u_stat - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            effect_size_r = z / np.sqrt(n1 + n2)

            pairwise.append({
                'Group1': GROUP_NAMES[i], 'Group2': GROUP_NAMES[j],
                'Group1_N': n1, 'Group2_N': n2,
                'Group1_Mean': round(d1.mean(), 6), 'Group2_Mean': round(d2.mean(), 6),
                'Group1_SD': round(d1.std(ddof=1), 6), 'Group2_SD': round(d2.std(ddof=1), 6),
                'Mann_Whitney_U': u_stat, 'Raw_P_Value': raw_p, 'Effect_Size_r': round(effect_size_r, 4),
            })
            raw_p_values.append(raw_p)

    if raw_p_values:
        rejected, corrected_p, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')
        for result, p_corr, rej in zip(pairwise, corrected_p, rejected):
            result['Bonferroni_P_Value'] = p_corr
            result['Significant_Bonferroni'] = 'Yes' if rej else 'No'
            result['Significant_Uncorrected'] = 'Yes' if result['Raw_P_Value'] < 0.05 else 'No'

        print(f"Bonferroni-significant comparisons: {sum(rejected)}/{len(raw_p_values)}")

    return descriptive_df, pd.DataFrame(pairwise), kw_stat, kw_p


def add_significance_brackets(ax, subgroup_data):
    all_data = [v for k in GROUP_KEYS for v in get_group_values(subgroup_data, k).tolist()]
    if not all_data:
        return

    max_val = max(all_data)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_levels = [0.1, 0.175, 0.25, 0.325, 0.4, 0.475, 0.55, 0.625, 0.7, 0.775]

    pairs, raw_p_values = [], []
    for i in range(len(GROUP_KEYS)):
        for j in range(i + 1, len(GROUP_KEYS)):
            d1 = get_group_values(subgroup_data, GROUP_KEYS[i])
            d2 = get_group_values(subgroup_data, GROUP_KEYS[j])
            p = mannwhitneyu(d1, d2, alternative='two-sided')[1] if len(d1) and len(d2) else 1.0
            pairs.append((i, j))
            raw_p_values.append(p)

    corrected_p = multipletests(raw_p_values, alpha=0.05, method='bonferroni')[1] if raw_p_values else raw_p_values

    for idx, ((i, j), p) in enumerate(zip(pairs, corrected_p)):
        if idx >= len(bracket_levels):
            continue
        x1, x2 = i + 1, j + 1
        bracket_y = max_val + y_range * bracket_levels[idx]
        tick = y_range * 0.008

        ax.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1.0)
        ax.plot([x1, x1], [bracket_y - tick, bracket_y], 'k-', linewidth=1.0)
        ax.plot([x2, x2], [bracket_y - tick, bracket_y], 'k-', linewidth=1.0)
        ax.text((x1 + x2) / 2, bracket_y + y_range * 0.01, get_significance_asterisks(p),
                ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_boxplot(subgroup_data, filename):
    data_to_plot, colors, labels = [], [], []

    for name, key in zip(GROUP_NAMES, GROUP_KEYS):
        data = get_group_values(subgroup_data, key)
        data_to_plot.append(data)
        colors.append(COLORS[name])
        display_name = name.replace(' (incl Likely AL)', '\n(incl Likely AL)')
        labels.append(f'{display_name}\n(n={len(data)})')

    fig, ax = plt.subplots(figsize=(16, 12))
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.8 if element != 'fliers' else 1.0)

    for i, data in enumerate(data_to_plot):
        if len(data) > 0:
            ax.scatter([i + 1], [data.mean()], marker='x', s=150, c='black', linewidth=1.5, zorder=10)

    add_significance_brackets(ax, subgroup_data)

    ax.set_ylim(ax.get_ylim()[0], 38.5)
    ax.set_title('Subgroup Analysis: Mean SNOMED Terms IC per Patient\n'
                 '(Amyloidosis Pre-Diagnosis vs Control Pre-Reference)',
                 fontsize=20, fontweight='bold', pad=35)
    ax.set_ylabel('Mean SNOMED Terms IC per Patient', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=25, ha='right', fontsize=13)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_density(subgroup_data, filename):
    fig, ax = plt.subplots(figsize=(16, 10))
    max_density = 0

    for name, key in zip(GROUP_NAMES, GROUP_KEYS):
        data = get_group_values(subgroup_data, key)
        if len(data) <= 5:
            continue

        color = COLORS[name]
        counts, _, _ = ax.hist(data, bins=30, density=True, alpha=0.3, color=color,
                                label=f'{name} (n={len(data)})')
        max_density = max(max_density, counts.max())

        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 300)
        density = kde(x_range)
        ax.plot(x_range, density, color=color, linewidth=3.5, label=f'{name} Density')
        max_density = max(max_density, density.max())
        ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2.5, alpha=0.9)

    ax.set_title('Subgroup Analysis: Mean SNOMED Terms IC per Patient\n'
                 '(Amyloidosis Pre-Diagnosis vs Control Pre-Reference)',
                 fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel('Mean SNOMED Terms IC per Patient', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.grid(True, alpha=0.3)
    if max_density > 0:
        ax.set_ylim(0, max_density * 1.05)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def export_results(subgroup_data, descriptive, pairwise, kw_stat, kw_p, patient_groups, n_controls, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary = pd.DataFrame([
            ['Comparison', 'Amyloidosis Pre-Diagnosis vs Control Pre-Reference'],
            ['Controls (n)', n_controls],
            ['Localised Cutaneous (n)', len(patient_groups['localised_cutaneous'])],
            ['AL Amyloidosis incl Likely AL (n)', len(patient_groups['al_amyloidosis_incl_likely'])],
            ['ATTR Amyloidosis (n)', len(patient_groups['attr_amyloidosis'])],
            ['Other Amyloidosis (n)', len(patient_groups['other_amyloidosis'])],
            ['Kruskal-Wallis H', kw_stat if kw_stat is not None else 'Not performed'],
            ['Kruskal-Wallis p', kw_p if kw_p is not None else 'Not performed'],
        ])
        summary.to_excel(writer, sheet_name='Summary', index=False, header=False)
        descriptive.to_excel(writer, sheet_name='Descriptive_Statistics', index=False)
        if not pairwise.empty:
            pairwise.to_excel(writer, sheet_name='Pairwise_Comparisons', index=False)

        assignments = [
            {'Patient_ID': pid, 'Subgroup': next(n for n, k in GROUP_KEY_BY_NAME.items() if k == key)}
            for key, ids in patient_groups.items() for pid in ids
        ]
        pd.DataFrame(assignments).to_excel(writer, sheet_name='Patient_Assignments', index=False)

        for key in GROUP_KEYS:
            if key in subgroup_data and len(subgroup_data[key]) > 0:
                sheet_name = key.replace('_', ' ').title()[:31]
                subgroup_data[key].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Exported: {output_file}")


def main():
    cache_key = get_cache_key(PATIENT_DATA_FILE, LAB_DATA_FILES)

    patient_groups = load_patient_classifications()
    amyl_visits, control_visits, control_ids = load_visit_level_ic_data(cache_key)
    patient_ic_df = calculate_patient_ic(amyl_visits, control_visits)
    subgroup_data = build_subgroup_data(patient_groups, patient_ic_df, control_ids)

    descriptive, pairwise, kw_stat, kw_p = run_statistical_analysis(subgroup_data)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_boxplot(subgroup_data, os.path.join(OUTPUT_DIR, f"subgroup_ic_boxplot_{timestamp}.png"))
    plot_density(subgroup_data, os.path.join(OUTPUT_DIR, f"subgroup_ic_density_{timestamp}.png"))

    output_file = os.path.join(OUTPUT_DIR, f"subgroup_ic_analysis_{timestamp}.xlsx")
    export_results(subgroup_data, descriptive, pairwise, kw_stat, kw_p,
                    patient_groups, len(control_ids), output_file)

    return {
        'patient_groups': patient_groups, 'subgroup_data': subgroup_data,
        'statistics': descriptive, 'comparisons': pairwise,
        'kruskal_wallis': {'H': kw_stat, 'p': kw_p}, 'output_file': output_file,
    }


if __name__ == "__main__":
    main()
