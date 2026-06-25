"""
Column / leakage / predictor policy for the Shopify SKU-sales panel (Experiment 3).

Reuses the GENERIC, dataset-agnostic helpers from clean_forecast_demand (winsor cap,
log1p transform, temporal split, imputed+missingness variant, Croston) so the two
experiments share one tested transform path. This module only declares the SKU panel's
column policy.
"""

import numpy as np
import pandas as pd

# re-export the shared, dataset-agnostic machinery (single source of truth)
from utils.clean.clean_forecast_demand import (  # noqa: F401
    fit_winsor_cap, to_log_target, invert_log_target, temporal_split,
    build_imputed_variant, _croston_forecast,
)

TARGET_COL = "units_sold"

# Same-week / realized-after-order signals -> leakage (known only at/after week t).
LEAKAGE_SIGNAL_COLS = ["n_lines", "avg_unit_price"]
# Identifier (the raw SKU string encodes product identity).
ID_LEAKAGE_COLS = ["sku"]
# Calendar keys not used as predictors.
NON_PREDICTOR_KEYS = ["week_start", "iso_year"]

# Allowed predictors (all leak-safe: lagged sales, calendar, static SKU attrs).
LAG_PREDICTORS = [
    "units_lag_1", "units_lag_2", "units_lag_4", "units_lag_8", "units_lag_12",
    "units_roll_mean_4", "units_roll_mean_8", "units_roll_std_4",
    "n_lines_lag1", "weeks_since_last_sale",
]
CALENDAR_PREDICTORS = ["iso_week", "month", "quarter", "week_sin", "week_cos"]
STATIC_CATEGORICAL = ["manufacturer", "status"]
STATIC_NUMERIC = ["sku_matches_part", "is_project", "unit_cost_current", "bom_indegree"]

# Leak-safe naive-baseline columns (already shifted in the panel).
BASELINE_LASTVALUE_COL = "units_lag_1"
BASELINE_ROLLMEAN_COL = "units_roll_mean_4"
BASELINE_CROSTON_COL = "croston_pred"


def clean_sku_sales(data):
    """Apply the SKU-panel policy; return (clean_df, predictors, cat_cols, num_cols).

    Adds a leak-safe per-SKU Croston baseline column computed on history < week t."""
    df = data.copy().sort_values(["sku", "week_start"]).reset_index(drop=True)

    # leak-safe Croston per SKU (uses the shared online implementation)
    preds = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("sku").groups.items():
        pos = df.index.get_indexer(idx)
        preds[pos] = _croston_forecast(df[TARGET_COL].values[pos])
    df[BASELINE_CROSTON_COL] = preds

    predictors = LAG_PREDICTORS + CALENDAR_PREDICTORS + STATIC_CATEGORICAL + STATIC_NUMERIC
    leakage_all = set(LEAKAGE_SIGNAL_COLS + ID_LEAKAGE_COLS + NON_PREDICTOR_KEYS + [TARGET_COL])
    assert leakage_all.isdisjoint(predictors), "leakage column leaked into predictors!"
    missing = [c for c in predictors if c not in df.columns]
    assert not missing, f"declared predictors missing from panel: {missing}"

    cat_cols = []
    for c in STATIC_CATEGORICAL:
        df[c] = df[c].astype("object")
        cat_cols.append(c)
    num_cols = [c for c in predictors if c not in cat_cols]

    keep = list(dict.fromkeys(
        ["week_start", TARGET_COL] + predictors + [BASELINE_CROSTON_COL]))
    clean_df = df[keep].copy()

    print(f"[clean_sku_sales] Predictors: {len(predictors)} "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"[clean_sku_sales] Dropped leakage: same-week {LEAKAGE_SIGNAL_COLS} + "
          f"id {ID_LEAKAGE_COLS}")
    return clean_df, predictors, cat_cols, num_cols
