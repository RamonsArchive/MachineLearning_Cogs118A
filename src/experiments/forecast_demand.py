"""
EXPERIMENT 1 - ModalAI Part/Component Demand Forecasting (REGRESSION)

Goal: predict demand_qty for a (part, week) from features known strictly BEFORE that
week. Mirrors experiments/parkinsons.py (GridSearchCV-tuned XGBoost / Random Forest /
Neural Net, 3 splits x 3 trials, JSON + per-model *_report.txt + plots + comparison),
adapted for this dataset's specifics:

  * TEMPORAL split (NOT shuffled): the earliest 20/50/80% of weeks are TRAIN, the
    remainder (strictly later) is TEST -> splits named 20_80 / 50_50 / 80_20.
  * 3 trials/split vary the seed (model init + CV shuffling); the split itself is
    deterministic, so trial spread is small (reported as mean +/- std).
  * LEAKAGE: every same-week signal, identifiers/SKU-name, and cum_zero_share
    (expanding mean incl. week t) are excluded - see clean_forecast_demand.py.
  * NULLS: rows kept. XGBoost gets NaN natively; RF/NN get median-impute +
    missingness-indicator (fit on TRAIN only).
  * TARGET: winsorize recount spikes + log1p; all reported metrics are inverted to
    real units. Also reports MAE on non-zero part-weeks and naive baselines
    (last-value, 4-week moving-average, Croston) as a fair intermittent-demand floor.

Grids are right-sized for the 126k-row panel (the parkinsons grids were tuned for
~6k rows; running those 192-combo grids x 27 GridSearchCV fits here is intractable).
Set env FORECAST_SMOKE=1 for a fast 1-split/1-trial/tiny-grid pipeline check.
"""

import os
import sys
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)

from utils.load.load_forecast_demand import load_forecast_demand_data
from utils.clean.clean_forecast_demand import (
    clean_forecast_demand, fit_winsor_cap, to_log_target, invert_log_target,
    temporal_split, build_imputed_variant,
    LEAKAGE_SIGNAL_COLS, ID_LEAKAGE_COLS, CURRENT_WEEK_LEAKAGE_COLS,
    DEAD_NULL_COLS, CONSTANT_COLS,
    BASELINE_LASTVALUE_COL, BASELINE_ROLLMEAN_COL, BASELINE_CROSTON_COL,
)
from utils.eda.eda_forecast_demand import eda_forecast_demand

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment

from graphs.forecast_demand_plots import (
    plot_forecast_demand_boosting_summary,
    plot_forecast_demand_random_forest_summary,
    plot_forecast_demand_neural_network_summary,
    plot_forecast_demand_model_comparison,
)

RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"
TARGET_COL = "demand_qty"
LOG_TARGET_COL = "y_log"

SMOKE = os.environ.get("FORECAST_SMOKE", "0") == "1"


# ==========================================
# Metric helpers (real units, after inverting log1p)
# ==========================================
def score_regression(y_true_real, y_pred_real):
    """RMSE/MAE/R2/MSE on real units + MAE restricted to non-zero-demand weeks."""
    y_true_real = np.asarray(y_true_real, dtype=float)
    y_pred_real = np.asarray(y_pred_real, dtype=float)
    mse = mean_squared_error(y_true_real, y_pred_real)
    out = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_real, y_pred_real)),
        "r2": float(r2_score(y_true_real, y_pred_real)),
    }
    nz = y_true_real > 0
    out["mae_nonzero"] = (
        float(mean_absolute_error(y_true_real[nz], y_pred_real[nz])) if nz.any()
        else float("nan")
    )
    return out


def _record_from_result(result, trial, train_df, test_df, predictors, cap, want_importance):
    """Invert log predictions -> real units, recompute metrics, pack a trial record.

    Predictions are clipped to [0, cap]: the target is winsorized at `cap`, so a model
    (esp. the NN, whose log-space output is unbounded) predicting beyond it is
    meaningless and would otherwise explode through expm1. Keeps y_test/y_pred (real
    units) on the in-memory record for plotting; these are stripped before the JSON
    dump (panel test sets are ~25k-100k rows)."""
    model = result["model"]

    y_test_real = invert_log_target(result["y_test"])              # already <= cap
    y_pred_real = np.clip(invert_log_target(result["y_pred"]), 0, cap)
    test_metrics = score_regression(y_test_real, y_pred_real)

    # real-units TRAIN rmse for an honest overfitting check (re-predict on train)
    y_train_real = invert_log_target(train_df[LOG_TARGET_COL].values)
    y_train_pred_real = np.clip(invert_log_target(model.predict(train_df[predictors])), 0, cap)
    train_rmse = float(np.sqrt(mean_squared_error(y_train_real, y_train_pred_real)))

    rec = {
        "trial": trial,
        "best_params": result["best_params"],
        "cv_scoring": result["cv_scoring"],
        "cv_train_score": result["cv_train_score"],
        "cv_val_score": result["cv_val_score"],
        "test_metrics": test_metrics,
        "train_metrics": {"rmse": train_rmse},
        "y_test": y_test_real.tolist(),
        "y_pred": y_pred_real.tolist(),
    }
    if want_importance and result.get("feature_importances") is not None:
        rec["feature_importances"] = np.asarray(result["feature_importances"]).tolist()
        rec["feature_names"] = result.get("feature_names")
    return rec


# ==========================================
# Naive baselines (deterministic per split)
# ==========================================
def compute_baselines(test_df, cap):
    """Evaluate naive forecasts on the same held-out weeks & winsorized truth as the
    models. Baseline preds are clipped to [0, cap] (same denoising as the target)."""
    y_true = np.clip(test_df[TARGET_COL].values.astype(float), 0, cap)
    out = {}
    for name, col in [
        ("last_value", BASELINE_LASTVALUE_COL),
        ("moving_avg_4w", BASELINE_ROLLMEAN_COL),
        ("croston", BASELINE_CROSTON_COL),
    ]:
        pred = np.clip(np.nan_to_num(test_df[col].values.astype(float), nan=0.0), 0, cap)
        out[name] = score_regression(y_true, pred)
    return out


# ==========================================
# Model grids (right-sized for 126k rows)
# ==========================================
def generate_boosting(train_df, test_df, rs, predictors):
    grid = ({"model__n_estimators": [200], "model__max_depth": [6]} if SMOKE else {
        "model__n_estimators": [50, 100, 400],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [1, 2, 3, 6, 10],
        "model__subsample": [0.5, 0.8, 1.0],
        "model__reg_lambda": [0.01, 0.1, 1.0, 10.0],
    })
    return run_boosting_experiment(
        train_df=train_df, test_df=test_df, predictors=predictors,
        target_col=LOG_TARGET_COL, problem_type="regression",
        random_state=rs, param_grid=grid, scoring="neg_mean_squared_error",
    )


def generate_random_forest(train_df, test_df, rs, predictors):
    grid = ({"model__n_estimators": [100], "model__max_depth": [12]} if SMOKE else {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [1, 2, 3, 12, 20, None],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt"],
    })
    return run_random_forest_experiment(
        train_df=train_df, test_df=test_df, predictors=predictors,
        target_col=LOG_TARGET_COL, problem_type="regression",
        random_state=rs, param_grid=grid, scoring="neg_mean_squared_error",
    )


def generate_neural_network(train_df, test_df, rs, predictors):
    grid = ({"model__hidden_layer_sizes": [(64, 32)], "model__alpha": [1e-3],
             "model__learning_rate_init": [0.01], "model__batch_size": [256]}
            if SMOKE else {
        "model__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
        "model__alpha": [1e-3, 1e-2, 1e-1],
        "model__learning_rate_init": [0.01, 0.05],
        "model__batch_size": [16, 32, 64, 128, 256],
    })
    return run_neural_net_experiment(
        train_df=train_df, test_df=test_df, predictors=predictors,
        target_col=LOG_TARGET_COL, problem_type="regression",
        random_state=rs, param_grid=grid, scoring="neg_mean_squared_error",
    )


# ==========================================
# Main
# ==========================================
def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: MODALAI DEMAND FORECAST (REGRESSION)")
    if SMOKE:
        print(">>> SMOKE MODE (1 split x 1 trial, tiny grids) <<<")
    print("=" * 60)

    # 1. Load + clean (encodes leakage / null / predictor policy)
    data = load_forecast_demand_data(curr_dir)
    clean_df, predictors, cat_cols, num_cols = clean_forecast_demand(data)

    # 2. EDA
    eda_forecast_demand(clean_df)

    print(f"\n[forecast_demand] Target: {TARGET_COL} (winsorized + log1p)")
    print(f"[forecast_demand] Predictors: {len(predictors)} "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # 3. Config: time fractions -> earliest X% of weeks are TRAIN
    split_configs = {"80_20": 0.80} if SMOKE else {
        "20_80": 0.20,  # earliest 20% of weeks train
        "50_50": 0.50,
        "80_20": 0.80,
    }
    n_trials = 1 if SMOKE else 3

    results = {BOOSTING_NAME: {}, RANDOM_FOREST_NAME: {}, NEURAL_NETWORK_NAME: {}}
    baselines = {}        # baseline-major: {baseline: {split: metrics}}
    split_info = {}

    # 4. Run
    for split_name, train_frac in split_configs.items():
        train_df, test_df, info = temporal_split(clean_df, train_frac)
        split_info[split_name] = info
        print(f"\n{'='*60}\nSPLIT {split_name}  "
              f"(train<= {info['cutoff_week']}: {info['train_rows']} rows / "
              f"{info['n_train_weeks']} wks | test: {info['test_rows']} rows / "
              f"{info['n_test_weeks']} wks)\n{'='*60}")

        # FIX #1: skip splits whose TRAIN target has no demand events (zero variance).
        # The component demand signal (stock_history / build_storage) only starts
        # 2025-12-09, so early-window splits train on an all-zero demand_qty. With an
        # all-zero train target the winsor cap (train p99.9) collapses to 0, log1p maps
        # everything to 0, and the model "perfectly predicts" zeros -> fake RMSE=0/R2=1.
        # There is no signal to learn here, so we skip the split entirely rather than
        # emit misleading metrics.
        n_train_demand = int((train_df[TARGET_COL] > 0).sum())
        if n_train_demand == 0:
            info["skipped"] = "no demand-bearing train rows (signal starts after cutoff)"
            print(f"[split] SKIP {split_name}: 0 demand>0 rows in train "
                  f"(demand signal starts after {info['cutoff_week']}); no signal to learn.")
            continue
        info["skipped"] = False
        print(f"[split] train demand>0 rows = {n_train_demand}")

        # winsor cap from TRAIN only -> log target on both
        cap = fit_winsor_cap(train_df[TARGET_COL].values)
        train_df = train_df.copy(); test_df = test_df.copy()
        train_df[LOG_TARGET_COL] = to_log_target(train_df[TARGET_COL].values, cap)
        test_df[LOG_TARGET_COL] = to_log_target(test_df[TARGET_COL].values, cap)
        print(f"[split] winsor cap (train p99.9) = {cap:.1f}")

        # imputed + missingness-indicator variant for RF/NN (fit on train)
        train_imp, test_imp, predictors_imp = build_imputed_variant(
            train_df, test_df, num_cols, cat_cols)

        # naive baselines (deterministic -> compute once per split); store baseline-major
        for bname, bmetrics in compute_baselines(test_df, cap).items():
            baselines.setdefault(bname, {})[split_name] = bmetrics

        for m in results:
            results[m][split_name] = []

        for trial in range(n_trials):
            rs = RANDOM_STATE + trial
            print(f"\n----- Trial {trial + 1}/{n_trials} (seed={rs}) -----")

            print(">>> XGBoost (NaN kept natively)...")
            xgb_res = generate_boosting(train_df, test_df, rs, predictors)
            results[BOOSTING_NAME][split_name].append(
                _record_from_result(xgb_res, trial, train_df, test_df, predictors, cap, True))

            print(">>> Random Forest (median-impute + missingness flag)...")
            rf_res = generate_random_forest(train_imp, test_imp, rs, predictors_imp)
            results[RANDOM_FOREST_NAME][split_name].append(
                _record_from_result(rf_res, trial, train_imp, test_imp, predictors_imp, cap, True))

            print(">>> Neural Network (impute + scale + missingness flag)...")
            nn_res = generate_neural_network(train_imp, test_imp, rs, predictors_imp)
            results[NEURAL_NETWORK_NAME][split_name].append(
                _record_from_result(nn_res, trial, train_imp, test_imp, predictors_imp, cap, False))

    # 5. Save JSON (strip large y arrays; keep metrics/params/importances + meta)
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_results = copy.deepcopy(results)
    for m in json_results:
        for s in json_results[m]:
            for rec in json_results[m][s]:
                rec.pop("y_test", None)
                rec.pop("y_pred", None)
    payload = {
        "_meta": {
            "task": "regression: predict weekly demand_qty per part",
            "target": TARGET_COL,
            "target_transform": "winsorize @ train p99.9 then log1p; metrics inverted",
            "split_type": "temporal (test weeks strictly later than train)",
            "splits": split_info,
            "n_trials": n_trials,
            "n_predictors": len(predictors),
            "predictors": predictors,
            "leakage_excluded": {
                "same_week_signals": LEAKAGE_SIGNAL_COLS,
                "identifiers_sku_name": ID_LEAKAGE_COLS,
                "current_week_leak": CURRENT_WEEK_LEAKAGE_COLS,
                "dropped_high_null": DEAD_NULL_COLS,
                "dropped_constant": CONSTANT_COLS,
            },
            "null_policy": "keep rows; XGB native NaN; RF/NN median-impute + missingness flag",
            "smoke": SMOKE,
        },
        "baselines": baselines,
        "models": json_results,
    }
    out_path = os.path.join(results_dir, "forecast_demand_all_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[forecast_demand] Saved results -> {out_path}")

    # 6. Plots + reports
    plots_base = os.path.join(curr_dir, "../..", "plots/forecast_demand_plots/results")
    plot_forecast_demand_boosting_summary(
        results[BOOSTING_NAME], os.path.join(plots_base, "boosting"))
    plot_forecast_demand_random_forest_summary(
        results[RANDOM_FOREST_NAME], os.path.join(plots_base, "random_forest"))
    plot_forecast_demand_neural_network_summary(
        results[NEURAL_NETWORK_NAME], os.path.join(plots_base, "neural_network"))
    plot_forecast_demand_model_comparison(
        results, baselines, os.path.join(plots_base, "comparison"))

    # 7. Final summary table
    print("\n" + "=" * 78)
    print("FINAL SUMMARY  -  Test RMSE / R2 / MAE (mean +/- std over trials)")
    print("=" * 78)
    splits = list(results[BOOSTING_NAME].keys())  # only splits that actually ran (FIX #1)
    print(f"{'Model':<16}{'Split':<8}{'RMSE':<18}{'R2':<16}{'MAE':<16}")
    print("-" * 78)
    for m in results:
        for s in splits:
            tr = results[m][s]
            rmse = np.array([t["test_metrics"]["rmse"] for t in tr])
            r2 = np.array([t["test_metrics"]["r2"] for t in tr])
            mae = np.array([t["test_metrics"]["mae"] for t in tr])
            print(f"{m:<16}{s:<8}"
                  f"{rmse.mean():>7.3f}+/-{rmse.std():<6.3f} "
                  f"{r2.mean():>6.3f}+/-{r2.std():<5.3f} "
                  f"{mae.mean():>6.3f}+/-{mae.std():<5.3f}")
    print("-" * 78)
    print("Naive baselines (RMSE by split):")
    for bname in ["last_value", "moving_avg_4w", "croston"]:
        row = "  " + f"{bname:<14}"
        for s in splits:
            row += f"{baselines[bname][s]['rmse']:>8.3f}  "
        print(row)
    print("\n[forecast_demand] EXPERIMENT 1 COMPLETE.")


if __name__ == "__main__":
    main()
