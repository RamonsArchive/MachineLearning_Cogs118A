"""
Cleaning + leakage control + feature policy for the ModalAI demand panel.

This module is where ALL of the dataset's modeling rules live, so the experiment
file stays a thin orchestrator (mirrors clean_parkinsons.py). Everything here is
driven by forecasting/docs/BUSINESS_IMPLICATIONS_AND_ML_FEASIBILITY.md and the EDA
SUMMARY.md null/leakage policy.

Key decisions encoded here
--------------------------
1. LEAKAGE: demand_qty = max(sh_outflow_qty, build_consumed_qty), so every
   contemporaneous (same-week) signal leaks the target and is dropped from
   predictors. Identifier / SKU-name columns (part_id, part_name, mpn) are dropped
   because the SKU name encodes the project (notes.txt) -> memorization.
   cum_zero_share is dropped too: build_god_dataset.py computes it with
   `.expanding().mean()` over (demand_qty==0) which INCLUDES the current week,
   so it leaks whether this week's demand is zero. Verified, not shifted -> drop.

2. NULL POLICY (EDA SUMMARY.md S6): keep null ROWS (+ missingness flag for the
   imputed variant); drop a COLUMN only when >=~80% null AND non-target/non-leakage.
   That removes low_stock_level (99.1% null) and lead_time_days (98.2% null).
   Constant columns (kanban, virtual: 1 unique value) carry no signal -> dropped.

3. ALLOWED PREDICTORS ONLY: lagged/rolling demand, calendar, and static part
   attributes. Static attrs are *current* snapshots (a known backtesting caveat,
   flagged in the report) but kept per the task spec.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Column policy
# ----------------------------------------------------------------------------

# Same-week signals that define / co-move with the target -> leakage.
LEAKAGE_SIGNAL_COLS = [
    "sh_outflow_qty", "build_consumed_qty", "sh_net_delta", "sh_inflow_qty",
    "sh_movement_events", "build_events", "shopify_ordered_qty",
    "shopify_unfulfilled_qty", "on_hand_eow",
]
# Identifiers / SKU-name leakage (name encodes the project).
ID_LEAKAGE_COLS = ["part_id", "part_name", "mpn"]
# Includes current week (expanding mean over demand==0) -> target leakage.
CURRENT_WEEK_LEAKAGE_COLS = ["cum_zero_share"]
# >=80% null, non-target, non-leakage -> drop per null policy.
DEAD_NULL_COLS = ["low_stock_level", "lead_time_days"]
# Single unique value -> zero variance.
CONSTANT_COLS = ["kanban", "virtual"]
# Calendar keys not in the allowed predictor list (iso_year is a raw time index
# -> trivial trend leakage across the temporal split; week_start is the split key).
NON_PREDICTOR_KEYS = ["week_start", "iso_year"]

TARGET_COL = "demand_qty"

# Allowed predictors (after the drops above).
LAG_ROLL_PREDICTORS = [
    "demand_lag_1", "demand_lag_2", "demand_lag_4", "demand_lag_8", "demand_lag_12",
    "demand_roll_mean_4", "demand_roll_mean_8", "demand_roll_std_4",
]
CALENDAR_PREDICTORS = ["iso_week", "month", "quarter", "week_sin", "week_cos"]
STATIC_CATEGORICAL = ["manufacturer", "status", "archived", "subassembly"]
STATIC_NUMERIC = [
    "is_project", "unit_cost_current", "price_current", "current_stock",
    "bom_component_count", "on_order_current", "moq", "order_multiple", "num_vendors",
]

# Leak-safe naive-baseline columns we keep alongside predictors (shifted in the panel).
BASELINE_LASTVALUE_COL = "demand_lag_1"      # last-value / naive-1 forecast
BASELINE_ROLLMEAN_COL = "demand_roll_mean_4"  # 4-week moving-average forecast
BASELINE_CROSTON_COL = "croston_pred"         # Croston's method (computed below)


def _croston_forecast(demand, alpha=0.1):
    """
    Leak-safe Croston's method for a single part's weekly demand series.

    For each week t, returns the forecast that depends ONLY on demand strictly
    before t (so it can be evaluated on held-out future weeks without leakage).
    Croston smooths nonzero demand size (z) and the inter-demand interval (p)
    separately; forecast = z / p. Before the first nonzero demand the forecast is 0.
    """
    n = len(demand)
    out = np.zeros(n, dtype=float)
    z = None          # smoothed demand size
    p = None          # smoothed interval
    q = 1             # weeks since last nonzero demand
    for t in range(n):
        # forecast for week t uses only history < t
        out[t] = (z / p) if (z is not None and p is not None) else 0.0
        d = demand[t]
        if d > 0:
            if z is None:
                z, p = float(d), float(q)
            else:
                z = z + alpha * (d - z)
                p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    return out


def _add_croston(df):
    """Add a leak-safe per-part Croston forecast column."""
    df = df.sort_values(["part_id", "week_start"]).reset_index(drop=True)
    preds = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("part_id").groups.items():
        pos = df.index.get_indexer(idx)
        preds[pos] = _croston_forecast(df[TARGET_COL].values[pos])
    df[BASELINE_CROSTON_COL] = preds
    return df


def clean_forecast_demand(data):
    """
    Apply the leakage / null / predictor policy and return a tidy frame plus the
    predictor metadata the experiment needs.

    Returns
    -------
    clean_df : DataFrame with [week_start, demand_qty, <predictors>,
               <baseline cols>] only.
    predictors : list[str]   - allowed predictor columns
    cat_cols   : list[str]   - categorical predictors (object/bool -> one-hot)
    num_cols   : list[str]   - numeric predictors
    """
    df = data.copy()

    # --- compute Croston baseline BEFORE we drop part_id (needs the grain) ---
    df = _add_croston(df)

    # --- Restrict to demand-bearing weeks (drop the all-zero pre-onset history) ---
    # The component demand signal (stock_history / build_storage) only starts ~2025-12-08;
    # the ~41 weeks before it are identically zero for EVERY part -> no learnable signal,
    # and they collapse the early temporal splits (an all-zero train target makes the
    # winsor cap 0 and fakes RMSE=0 / R2=1). We keep weeks >= the first week with any
    # demand so all three splits train on real signal. Lags / rolling stats / Croston were
    # computed on the FULL history above, so their values at the onset weeks stay correct
    # after this row filter (e.g. demand_lag_1 at the onset week is the prior zero week).
    wk_totals = df.groupby("week_start")[TARGET_COL].sum()
    onset = wk_totals[wk_totals > 0].index.min()
    n_before = len(df)
    df = df[df["week_start"] >= onset].copy()
    print(f"[clean_forecast_demand] Restricted to demand-bearing weeks >= "
          f"{pd.Timestamp(onset).date()}: {len(df)} rows "
          f"({n_before - len(df)} pre-onset zero-rows dropped), "
          f"{df['week_start'].nunique()} weeks, {df['part_id'].nunique()} parts.")

    predictors = (
        LAG_ROLL_PREDICTORS + CALENDAR_PREDICTORS + STATIC_CATEGORICAL + STATIC_NUMERIC
    )

    # Guard: every declared predictor must exist; none may be a leakage column.
    leakage_all = set(
        LEAKAGE_SIGNAL_COLS + ID_LEAKAGE_COLS + CURRENT_WEEK_LEAKAGE_COLS
        + DEAD_NULL_COLS + CONSTANT_COLS + NON_PREDICTOR_KEYS
    )
    assert leakage_all.isdisjoint(predictors), "leakage column leaked into predictors!"
    missing = [c for c in predictors if c not in df.columns]
    assert not missing, f"declared predictors missing from panel: {missing}"

    # Normalize categoricals to clean string categories (object dtype, NaN -> stays NaN).
    cat_cols = []
    for c in STATIC_CATEGORICAL:
        df[c] = df[c].astype("object")
        cat_cols.append(c)

    num_cols = [c for c in predictors if c not in cat_cols]

    keep = (
        ["week_start", TARGET_COL]
        + predictors
        + [BASELINE_CROSTON_COL]  # lag_1 / roll_mean_4 already in predictors
    )
    keep = list(dict.fromkeys(keep))  # de-dup, preserve order
    clean_df = df[keep].copy()

    print(f"[clean_forecast_demand] Predictors: {len(predictors)} "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"[clean_forecast_demand] Dropped leakage signals: {LEAKAGE_SIGNAL_COLS}")
    print(f"[clean_forecast_demand] Dropped id/name leakage: {ID_LEAKAGE_COLS}")
    print(f"[clean_forecast_demand] Dropped current-week leakage: "
          f"{CURRENT_WEEK_LEAKAGE_COLS} (cum_zero_share = expanding mean incl. week t)")
    print(f"[clean_forecast_demand] Dropped >=80% null: {DEAD_NULL_COLS}; "
          f"constant: {CONSTANT_COLS}")

    return clean_df, predictors, cat_cols, num_cols


# ----------------------------------------------------------------------------
# Target transform  (winsorize -> log1p ; invert with expm1)
# ----------------------------------------------------------------------------

def fit_winsor_cap(y_train, q=0.999):
    """
    Cap derived from TRAIN only (no leakage). The +/-61k weekly spikes are recount
    corrections, not real demand (EDA SUMMARY.md), so we clip the extreme upper tail.
    Default q=0.999 removes the recount artifacts while preserving genuine large
    orders (train p99.9 ~ 1.6k vs max ~ 61k).
    """
    return float(np.quantile(y_train, q))


def to_log_target(y, cap):
    """Winsorize at `cap`, then log1p. Use for the model's training/eval target."""
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, cap))


def invert_log_target(y_log):
    """Invert log1p; clip tiny negative round-off to 0 (demand is non-negative)."""
    return np.clip(np.expm1(np.asarray(y_log, dtype=float)), 0, None)


# ----------------------------------------------------------------------------
# Temporal split  (NOT shuffled - test weeks strictly later than train weeks)
# ----------------------------------------------------------------------------

def temporal_split(clean_df, train_frac):
    """
    Split by time: the earliest `train_frac` of the distinct weeks become TRAIN,
    the remainder become TEST (strictly later). Returns (train_df, test_df, info).
    """
    weeks = np.sort(clean_df["week_start"].unique())
    n_train_weeks = max(1, int(round(len(weeks) * train_frac)))
    n_train_weeks = min(n_train_weeks, len(weeks) - 1)  # always keep >=1 test week
    cutoff = weeks[n_train_weeks - 1]

    train_df = clean_df[clean_df["week_start"] <= cutoff].copy()
    test_df = clean_df[clean_df["week_start"] > cutoff].copy()
    info = {
        "n_weeks": int(len(weeks)),
        "n_train_weeks": int(n_train_weeks),
        "n_test_weeks": int(len(weeks) - n_train_weeks),
        "cutoff_week": pd.Timestamp(cutoff).date().isoformat(),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    return train_df, test_df, info


# ----------------------------------------------------------------------------
# Imputed + missingness-indicator variant  (for Random Forest / Neural Net)
# ----------------------------------------------------------------------------

def build_imputed_variant(train_df, test_df, num_cols, cat_cols):
    """
    RF errors on NaN and NN needs finite scaled inputs, so per the EDA null policy we
    MEDIAN-IMPUTE numeric predictors AND add a `{col}_was_null` missingness indicator
    for every numeric column that has any missing value in TRAIN. Medians are fit on
    TRAIN ONLY (no leakage). Categorical NaN -> explicit "MISSING" level (one-hot picks
    it up). XGBoost does NOT use this variant - it keeps NaN natively.

    Returns (train_imp, test_imp, predictors_imp).
    """
    train_imp = train_df.copy()
    test_imp = test_df.copy()
    indicator_cols = []

    for c in num_cols:
        med = train_imp[c].median()
        if train_imp[c].isna().any():
            flag = f"{c}_was_null"
            train_imp[flag] = train_imp[c].isna().astype(int)
            test_imp[flag] = test_imp[c].isna().astype(int)
            indicator_cols.append(flag)
        train_imp[c] = train_imp[c].fillna(med)
        test_imp[c] = test_imp[c].fillna(med)

    for c in cat_cols:
        train_imp[c] = train_imp[c].astype("object").fillna("MISSING")
        test_imp[c] = test_imp[c].astype("object").fillna("MISSING")

    predictors_imp = num_cols + indicator_cols + cat_cols
    return train_imp, test_imp, predictors_imp
