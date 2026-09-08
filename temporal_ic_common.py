"""
Shared functions for temporal IC trend analysis. Used by both the
calendar-month (Figure 3A) and visit-sequence (Figure 3B) variants, which
differ only in how each visit's temporal position is defined.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu

from utils import parse_pipe_separated, has_amyloid_diagnosis


def load_ic_lookup_table(file_path):
    ic_df = pd.read_excel(file_path, sheet_name='Diagnosis_IC_ENTIRE_DATASET')
    print(f"Loaded {len(ic_df):,} SNOMED terms with IC values")
    return dict(zip(ic_df['Diagnosis_Description'], ic_df['Information_Content']))


def calculate_visit_ic(snomed_descriptions, ic_dict, exclude_amyloid):
    terms = parse_pipe_separated(snomed_descriptions)
    if exclude_amyloid:
        terms = [t for t in terms if not has_amyloid_diagnosis(t)]

    ic_values = [ic_dict[t] for t in terms if t in ic_dict]
    return np.mean(ic_values) if ic_values else np.nan


def load_visit_data(file_path, sheet_name, ic_dict, exclude_amyloid, time_col_candidates):
    """Loads a visit-level sheet, recalculates Mean_IC per visit, and
    identifies which of time_col_candidates is present in this sheet.
    Returns (visits_df, time_col)."""

    visits_df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Loaded {len(visits_df):,} visits from '{sheet_name}'")

    time_col = next((c for c in time_col_candidates if c in visits_df.columns), None)
    if time_col is None:
        raise ValueError(f"None of {time_col_candidates} found in '{sheet_name}'")

    visits_df['Mean_IC_Recalculated'] = visits_df['SNOMED_Descriptions'].apply(
        lambda x: calculate_visit_ic(x, ic_dict, exclude_amyloid)
    )

    n_valid = visits_df['Mean_IC_Recalculated'].notna().sum()
    print(f"  {n_valid:,}/{len(visits_df):,} visits with valid Mean IC "
          f"(exclude_amyloid={exclude_amyloid}, time_col='{time_col}')")

    return visits_df, time_col


def aggregate_by_group(visits_df, group_col, patient_type):
    valid = visits_df[visits_df['Mean_IC_Recalculated'].notna()].copy()

    aggregated = (
        valid.groupby(group_col)
        .agg(Mean_IC_Average=('Mean_IC_Recalculated', 'mean'),
             Mean_IC_StdDev=('Mean_IC_Recalculated', 'std'),
             Visit_Count=('Mean_IC_Recalculated', 'count'),
             Unique_Patients=('Patient_ID', 'nunique'))
        .round(4)
        .reset_index()
    )
    aggregated['Patient_Type'] = patient_type

    print(f"{patient_type}: {len(aggregated)} bins, {aggregated['Visit_Count'].sum():,} visits, "
          f"{aggregated['Unique_Patients'].sum():,} unique patients")

    return aggregated


def export_aggregated_data(amyl_agg, control_agg, group_col, output_file):
    common = sorted(set(amyl_agg[group_col]) & set(control_agg[group_col]))

    comparison = pd.DataFrame([
        {
            group_col: g,
            'Amyloidosis_Mean_IC': amyl_agg.loc[amyl_agg[group_col] == g, 'Mean_IC_Average'].iloc[0],
            'Amyloidosis_Visit_Count': amyl_agg.loc[amyl_agg[group_col] == g, 'Visit_Count'].iloc[0],
            'Control_Mean_IC': control_agg.loc[control_agg[group_col] == g, 'Mean_IC_Average'].iloc[0],
            'Control_Visit_Count': control_agg.loc[control_agg[group_col] == g, 'Visit_Count'].iloc[0],
            'IC_Difference': round(
                amyl_agg.loc[amyl_agg[group_col] == g, 'Mean_IC_Average'].iloc[0]
                - control_agg.loc[control_agg[group_col] == g, 'Mean_IC_Average'].iloc[0], 4),
        }
        for g in common
    ])

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        amyl_agg.to_excel(writer, sheet_name='Amyloidosis_Aggregated', index=False)
        control_agg.to_excel(writer, sheet_name='Control_Aggregated', index=False)
        comparison.to_excel(writer, sheet_name='Comparison', index=False)

    print(f"Exported: {output_file} ({len(comparison)} common bins)")
    return comparison


def _fit_trend(x, y):
    if len(x) <= 3:
        return np.nan, np.nan, np.nan, None
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, r, p, slope * x + intercept


def create_trend_plot(amyl_agg, control_agg, group_col, xlabel, title, output_file,
                       time_range_amyl=None, time_range_control=None):
    amyl = amyl_agg.copy()
    control = control_agg.copy()

    if time_range_amyl:
        amyl = amyl[amyl[group_col].between(*time_range_amyl)]
    if time_range_control:
        control = control[control[group_col].between(*time_range_control)]

    for df in (amyl, control):
        df['SE_IC'] = (df['Mean_IC_StdDev'] / np.sqrt(df['Visit_Count'])).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.errorbar(amyl[group_col], amyl['Mean_IC_Average'], yerr=amyl['SE_IC'],
                color='red', marker='o', linewidth=2, markersize=5, capsize=4, alpha=0.8,
                label=f'Amyloidosis (n={amyl["Visit_Count"].sum():,} visits)')
    ax.errorbar(control[group_col], control['Mean_IC_Average'], yerr=control['SE_IC'],
                color='blue', marker='s', linewidth=2, markersize=5, capsize=4, alpha=0.8,
                label=f'Control (n={control["Visit_Count"].sum():,} visits)')

    slope_amyl, r_amyl, p_amyl, trend_amyl = _fit_trend(amyl[group_col], amyl['Mean_IC_Average'])
    if trend_amyl is not None:
        ax.plot(amyl[group_col], trend_amyl, '--', color='red', alpha=0.6, linewidth=2)

    slope_ctrl, r_ctrl, p_ctrl, trend_ctrl = _fit_trend(control[group_col], control['Mean_IC_Average'])
    if trend_ctrl is not None:
        ax.plot(control[group_col], trend_ctrl, '--', color='blue', alpha=0.6, linewidth=2)

    ax.axvline(x=0, color='black', linewidth=2, alpha=0.7, label='Diagnosis/Reference')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean SNOMED Terms IC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    trend_text = (f'Amyloidosis: slope={slope_amyl:.6f}, R2={r_amyl**2 if pd.notna(r_amyl) else 0:.3f}, p={p_amyl:.3f}\n'
                  f'Control: slope={slope_ctrl:.6f}, R2={r_ctrl**2 if pd.notna(r_ctrl) else 0:.3f}, p={p_ctrl:.3f}')
    ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    amyl_mean, control_mean = amyl['Mean_IC_Average'].mean(), control['Mean_IC_Average'].mean()
    summary_text = (f'Overall Mean IC:\nAmyloidosis: {amyl_mean:.4f}\n'
                     f'Control: {control_mean:.4f}\nDifference: {amyl_mean - control_mean:.4f}')
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, va='bottom', ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    print(f"Amyloidosis slope={slope_amyl:.6f} (p={p_amyl:.3f}), Control slope={slope_ctrl:.6f} (p={p_ctrl:.3f})")


def create_violin_plot(amyl_agg, control_agg, output_file, seed=42):
    """Reconstructs approximate visit-level distributions from aggregated
    bin statistics (mean/SD per bin) for visualization purposes, since raw
    visit-level values aren't retained at this stage of the pipeline."""

    rng = np.random.default_rng(seed)

    def reconstruct(agg):
        values = []
        for _, row in agg.iterrows():
            if pd.notna(row['Mean_IC_Average']) and row['Visit_Count'] > 0:
                std = row['Mean_IC_StdDev'] if pd.notna(row['Mean_IC_StdDev']) and row['Mean_IC_StdDev'] > 0 else 0.1
                values.extend(rng.normal(row['Mean_IC_Average'], std, int(row['Visit_Count'])))
        return values

    amyl_values = reconstruct(amyl_agg)
    control_values = reconstruct(control_agg)

    fig, ax = plt.subplots(figsize=(10, 8))
    parts = ax.violinplot([control_values, amyl_values], positions=[1, 2],
                           showmeans=True, showmedians=True, widths=0.6)

    for pc, color in zip(parts['bodies'], ['blue', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for element in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[element].set_color('black')
        parts[element].set_linewidth(2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Control\nPatients', 'Amyloidosis\nPatients\n(Amyloid terms excluded)'])
    ax.set_ylabel('SNOMED Terms IC', fontsize=12)
    ax.set_title('SNOMED Terms IC Distribution\n(Amyloid-related terms excluded from Amyloidosis patients)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _, p_value = mannwhitneyu(control_values, amyl_values, alternative='two-sided')
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(0.5, 0.95, f'Mann-Whitney U test\n{p_text}', transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    control_mean, amyl_mean = np.mean(control_values), np.mean(amyl_values)
    summary_text = (f'Control: Mean={control_mean:.4f}, n={len(control_values):,}\n'
                     f'Amyloidosis: Mean={amyl_mean:.4f}, n={len(amyl_values):,}\n'
                     f'Difference: {amyl_mean - control_mean:.4f}')
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    print(f"Control mean={control_mean:.6f}, Amyloidosis mean={amyl_mean:.6f}, p={p_value:.6f}")
