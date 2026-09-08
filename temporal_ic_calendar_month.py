"""
Temporal IC trend by calendar time (Figure 3A). Visits are binned into
whole-month intervals relative to diagnosis (amyloidosis) or reference
midpoint (controls); amyloid-related SNOMED terms are excluded from
amyloidosis patients only.
"""

import os
from config import IC_EXCEL_FILE, OUTPUT_DIR
from temporal_ic_common import (
    load_ic_lookup_table, load_visit_data, aggregate_by_group,
    export_aggregated_data, create_trend_plot, create_violin_plot,
)

TIME_RANGE_AMYL = (-49, 49)
TIME_RANGE_CONTROL = (-25, 25)


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

    amyl_visits['Months_Rounded'] = amyl_visits[amyl_time_col].round().astype(int)
    control_visits['Months_Rounded'] = control_visits[control_time_col].round().astype(int)

    amyl_agg = aggregate_by_group(amyl_visits, 'Months_Rounded', 'Amyloidosis')
    control_agg = aggregate_by_group(control_visits, 'Months_Rounded', 'Control')

    export_aggregated_data(
        amyl_agg, control_agg, 'Months_Rounded',
        os.path.join(OUTPUT_DIR, "temporal_ic_calendar_month_aggregated.xlsx")
    )

    create_trend_plot(
        amyl_agg, control_agg, 'Months_Rounded',
        xlabel='Months from Diagnosis/Reference',
        title='Mean SNOMED Terms IC per Visit Temporal Trends\n(Control vs Amyloidosis | Amyloid Terms Filtered)',
        output_file=os.path.join(OUTPUT_DIR, "temporal_ic_calendar_month_plot.png"),
        time_range_amyl=TIME_RANGE_AMYL, time_range_control=TIME_RANGE_CONTROL,
    )

    create_violin_plot(amyl_agg, control_agg,
                        os.path.join(OUTPUT_DIR, "temporal_ic_calendar_month_violin.png"))


if __name__ == "__main__":
    main()
