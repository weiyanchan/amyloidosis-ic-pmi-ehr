"""
Temporal IC trend by visit sequence (Figure 3B). Each patient's visits are
re-indexed to a sequence number relative to their diagnosis/reference visit
(sequence 0), rather than calendar time.
"""

import os
import numpy as np
import pandas as pd
from config import IC_EXCEL_FILE, OUTPUT_DIR
from temporal_ic_common import (
    load_ic_lookup_table, load_visit_data, aggregate_by_group,
    export_aggregated_data, create_trend_plot, create_violin_plot,
)


def assign_visit_sequences(visits_df, time_column):
    """For each patient, re-indexes visits so the one closest to time 0
    (diagnosis/reference) becomes sequence 0, with negative/positive
    sequence numbers for earlier/later visits."""

    sequenced = []
    for _, patient_visits in visits_df.groupby('Patient_ID'):
        patient_visits = patient_visits.sort_values(time_column).copy()

        unique_times = np.sort(patient_visits[time_column].unique())
        diagnosis_idx = np.argmin(np.abs(unique_times))
        time_to_sequence = {t: i - diagnosis_idx for i, t in enumerate(unique_times)}

        patient_visits['Visit_Sequence'] = patient_visits[time_column].map(time_to_sequence)
        sequenced.append(patient_visits)

    result = pd.concat(sequenced, ignore_index=True)
    print(f"Visit sequence range: {result['Visit_Sequence'].min()} to {result['Visit_Sequence'].max()} "
          f"({result['Visit_Sequence'].nunique()} unique sequences)")
    return result


def main():
    ic_dict = load_ic_lookup_table(IC_EXCEL_FILE)

    amyl_visits, amyl_time_col = load_visit_data(
        IC_EXCEL_FILE, 'Amyloid_449_Visit_IC_ENTIRE', ic_dict,
        exclude_amyloid=True, time_col_candidates=['Months_From_Diagnosis']
    )
    control_visits, control_time_col = load_visit_data(
        IC_EXCEL_FILE, 'Control_NEW_VARIANCE_Visit', ic_dict,
        exclude_amyloid=False, time_col_candidates=['Months_From_Reference', 'Months_From_Diagnosis']
    )

    amyl_visits = assign_visit_sequences(amyl_visits, amyl_time_col)
    control_visits = assign_visit_sequences(control_visits, control_time_col)

    amyl_agg = aggregate_by_group(amyl_visits, 'Visit_Sequence', 'Amyloidosis')
    control_agg = aggregate_by_group(control_visits, 'Visit_Sequence', 'Control')

    export_aggregated_data(
        amyl_agg, control_agg, 'Visit_Sequence',
        os.path.join(OUTPUT_DIR, "temporal_ic_visit_sequence_aggregated.xlsx")
    )

    create_trend_plot(
        amyl_agg, control_agg, 'Visit_Sequence',
        xlabel='Visit Sequence (Relative to Diagnosis/Reference)',
        title='SNOMED Terms IC by Visit Sequence\n(Control vs Amyloidosis | Amyloid Terms Filtered)',
        output_file=os.path.join(OUTPUT_DIR, "temporal_ic_visit_sequence_plot.png"),
    )

    create_violin_plot(amyl_agg, control_agg,
                        os.path.join(OUTPUT_DIR, "temporal_ic_visit_sequence_violin.png"))


if __name__ == "__main__":
    main()
