import os
import pandas as pd

"""
ModalAI Part/Component Demand Forecasting Dataset
=================================================

Source: forecasting/ForcastingSets/unified/part_week_panel.csv (the "god dataset"),
copied into datasets/ForecastDemandSets/ so the original is never mutated.

GRAIN: one row per (part_id, ISO week) - weekly Mondays.
  70 weeks (2025-02-24 -> 2026-06-22), 1,809 active parts, 126,630 rows, 46 columns.

TASK: REGRESSION - predict demand_qty for a (part, week) from features known
      strictly BEFORE that week.

TARGET: demand_qty = max(sh_outflow_qty, build_consumed_qty)
  - StockHistory.quantity signed-delta outflow (realized internal consumption)
  - BuildStorage component pulls into builds
  This union avoids double counting overlapping internal + build-driven consumption.
  The intended Consumption table is empty in prod (0 rows), so demand is reconstructed.

  The target is heavily zero-inflated (~94% zeros) and heavy-tailed: weekly spikes of
  +/-61k are recount/correction artifacts, NOT real demand (see EDA SUMMARY.md). The
  experiment winsorizes and log1p-transforms the target (see clean_forecast_demand.py).
"""


def load_forecast_demand_data(curr_dir):
    """
    Load the ModalAI part-week demand panel.

    Returns:
        pd.DataFrame: full unified panel (126,630 x 46), unmodified.
    """
    data_path = os.path.join(
        curr_dir, "../../datasets/ForecastDemandSets/part_week_panel.csv"
    )
    data = pd.read_csv(data_path, parse_dates=["week_start"])

    print(
        f"[load_forecast_demand] Loaded {len(data)} part-weeks "
        f"({data['part_id'].nunique()} parts x {data['week_start'].nunique()} weeks)"
    )
    print(
        f"[load_forecast_demand] Week range: "
        f"{data['week_start'].min().date()} -> {data['week_start'].max().date()}"
    )
    return data
