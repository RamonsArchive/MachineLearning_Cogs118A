"""
EXPERIMENT 3 - ModalAI Shopify SKU-Sales Forecasting (REGRESSION)

Predict a SKU's weekly units_sold from features known strictly BEFORE that week. Same
methodology as Experiment 1 (temporal 20/50/80 splits x 3 trials, GridSearchCV XGB/RF/NN,
winsorize+log1p target inverted for metrics, naive baselines, MAE on non-zero weeks) on
the single-source Shopify panel that spans the full ~70 weeks (more history than the
component panel). Set FORECAST_SMOKE=1 for a fast 1-split/1-trial check.
"""
import os, sys, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.load.load_sku_sales import load_sku_sales_data
from utils.clean.clean_sku_sales import (
    clean_sku_sales, fit_winsor_cap, to_log_target, invert_log_target,
    temporal_split, build_imputed_variant, TARGET_COL,
    BASELINE_LASTVALUE_COL, BASELINE_ROLLMEAN_COL, BASELINE_CROSTON_COL,
)
from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment
from graphs.sku_sales_plots import (
    plot_sku_sales_boosting_summary, plot_sku_sales_random_forest_summary,
    plot_sku_sales_neural_network_summary, plot_sku_sales_model_comparison,
)

RANDOM_STATE = 42
BOOSTING, RF, NN = "boosting", "random_forest", "neural_network"
LOG_TARGET_COL = "y_log"
SMOKE = os.environ.get("FORECAST_SMOKE", "0") == "1"


def score_regression(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    mse = mean_squared_error(yt, yp)
    out = {"mse": float(mse), "rmse": float(np.sqrt(mse)),
           "mae": float(mean_absolute_error(yt, yp)), "r2": float(r2_score(yt, yp))}
    nz = yt > 0
    out["mae_nonzero"] = float(mean_absolute_error(yt[nz], yp[nz])) if nz.any() else float("nan")
    return out


def _record(result, trial, train_df, predictors, cap, want_importance):
    model = result["model"]
    yt = invert_log_target(result["y_test"])
    yp = np.clip(invert_log_target(result["y_pred"]), 0, cap)
    ytr = invert_log_target(train_df[LOG_TARGET_COL].values)
    ytrp = np.clip(invert_log_target(model.predict(train_df[predictors])), 0, cap)
    rec = {"trial": trial, "best_params": result["best_params"],
           "cv_scoring": result["cv_scoring"], "cv_train_score": result["cv_train_score"],
           "cv_val_score": result["cv_val_score"], "test_metrics": score_regression(yt, yp),
           "train_metrics": {"rmse": float(np.sqrt(mean_squared_error(ytr, ytrp)))},
           "y_test": yt.tolist(), "y_pred": yp.tolist()}
    if want_importance and result.get("feature_importances") is not None:
        rec["feature_importances"] = np.asarray(result["feature_importances"]).tolist()
        rec["feature_names"] = result.get("feature_names")
    return rec


def compute_baselines(test_df, cap):
    y = np.clip(test_df[TARGET_COL].values.astype(float), 0, cap)
    out = {}
    for name, col in [("last_value", BASELINE_LASTVALUE_COL),
                      ("moving_avg_4w", BASELINE_ROLLMEAN_COL),
                      ("croston", BASELINE_CROSTON_COL)]:
        p = np.clip(np.nan_to_num(test_df[col].values.astype(float), nan=0.0), 0, cap)
        out[name] = score_regression(y, p)
    return out


def _xgb(tr, te, rs, pred):
    grid = ({"model__n_estimators": [200], "model__max_depth": [4]} if SMOKE else {
        "model__n_estimators": [300, 600], "model__learning_rate": [0.03, 0.05],
        "model__max_depth": [3, 5], "model__subsample": [0.8],
        "model__reg_lambda": [1.0, 10.0]})
    return run_boosting_experiment(train_df=tr, test_df=te, predictors=pred,
        target_col=LOG_TARGET_COL, problem_type="regression", random_state=rs,
        param_grid=grid, scoring="neg_mean_squared_error")


def _rf(tr, te, rs, pred):
    grid = ({"model__n_estimators": [100], "model__max_depth": [10]} if SMOKE else {
        "model__n_estimators": [300], "model__max_depth": [6, 12, None],
        "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt"]})
    return run_random_forest_experiment(train_df=tr, test_df=te, predictors=pred,
        target_col=LOG_TARGET_COL, problem_type="regression", random_state=rs,
        param_grid=grid, scoring="neg_mean_squared_error")


def _nn(tr, te, rs, pred):
    grid = ({"model__hidden_layer_sizes": [(32, 16)], "model__alpha": [1e-2],
             "model__learning_rate_init": [0.01], "model__batch_size": [128]} if SMOKE else {
        "model__hidden_layer_sizes": [(32, 16), (64, 32)], "model__alpha": [1e-2, 1e-1],
        "model__learning_rate_init": [0.01], "model__batch_size": [128]})
    return run_neural_net_experiment(train_df=tr, test_df=te, predictors=pred,
        target_col=LOG_TARGET_COL, problem_type="regression", random_state=rs,
        param_grid=grid, scoring="neg_mean_squared_error")


def main():
    cd = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "=" * 60 + "\nEXPERIMENT 3: SHOPIFY SKU-SALES FORECAST (REGRESSION)")
    if SMOKE: print(">>> SMOKE MODE <<<")
    print("=" * 60)

    clean_df, predictors, cat_cols, num_cols = clean_sku_sales(load_sku_sales_data(cd))
    print(f"[sku_sales] Target: {TARGET_COL} (winsorized + log1p) | predictors: {len(predictors)}")

    split_cfg = {"80_20": 0.80} if SMOKE else {"20_80": 0.20, "50_50": 0.50, "80_20": 0.80}
    n_trials = 1 if SMOKE else 3
    results = {BOOSTING: {}, RF: {}, NN: {}}; baselines = {}; split_info = {}

    for sname, frac in split_cfg.items():
        tr, te, info = temporal_split(clean_df, frac); split_info[sname] = info
        print(f"\n{'='*60}\nSPLIT {sname} (train<= {info['cutoff_week']}: {info['train_rows']}r | "
              f"test {info['test_rows']}r)\n{'='*60}")
        if int((tr[TARGET_COL] > 0).sum()) == 0:
            info["skipped"] = True
            print(f"[split] SKIP {sname}: 0 sales in train."); continue
        info["skipped"] = False
        cap = fit_winsor_cap(tr[TARGET_COL].values)
        tr = tr.copy(); te = te.copy()
        tr[LOG_TARGET_COL] = to_log_target(tr[TARGET_COL].values, cap)
        te[LOG_TARGET_COL] = to_log_target(te[TARGET_COL].values, cap)
        print(f"[split] winsor cap = {cap:.1f}")
        tri, tei, pred_i = build_imputed_variant(tr, te, num_cols, cat_cols)
        for bn, bm in compute_baselines(te, cap).items():
            baselines.setdefault(bn, {})[sname] = bm
        for m in results: results[m][sname] = []
        for trial in range(n_trials):
            rs = RANDOM_STATE + trial
            print(f"\n--- Trial {trial+1}/{n_trials} (seed={rs}) ---")
            print(">>> XGBoost..."); r = _xgb(tr, te, rs, predictors)
            results[BOOSTING][sname].append(_record(r, trial, tr, predictors, cap, True))
            print(">>> Random Forest..."); r = _rf(tri, tei, rs, pred_i)
            results[RF][sname].append(_record(r, trial, tri, pred_i, cap, True))
            print(">>> Neural Network..."); r = _nn(tri, tei, rs, pred_i)
            results[NN][sname].append(_record(r, trial, tri, pred_i, cap, False))

    results_dir = os.path.join(cd, "..", "results"); os.makedirs(results_dir, exist_ok=True)
    jr = copy.deepcopy(results)
    for m in jr:
        for s in jr[m]:
            for rec in jr[m][s]: rec.pop("y_test", None); rec.pop("y_pred", None)
    payload = {"_meta": {"task": "regression: weekly Shopify units_sold per SKU",
                         "target": TARGET_COL, "split_type": "temporal", "splits": split_info,
                         "n_trials": n_trials, "predictors": predictors, "smoke": SMOKE},
               "baselines": baselines, "models": jr}
    with open(os.path.join(results_dir, "sku_sales_all_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[sku_sales] Saved results -> {os.path.join(results_dir, 'sku_sales_all_results.json')}")

    pb = os.path.join(cd, "../..", "plots/sku_sales_plots/results")
    plot_sku_sales_boosting_summary(results[BOOSTING], os.path.join(pb, "boosting"))
    plot_sku_sales_random_forest_summary(results[RF], os.path.join(pb, "random_forest"))
    plot_sku_sales_neural_network_summary(results[NN], os.path.join(pb, "neural_network"))
    plot_sku_sales_model_comparison(results, baselines, os.path.join(pb, "comparison"))

    print("\n" + "=" * 78 + "\nFINAL SUMMARY - Test RMSE / R2 / MAE (mean +/- std)\n" + "=" * 78)
    splits = list(results[BOOSTING].keys())
    print(f"{'Model':<16}{'Split':<8}{'RMSE':<18}{'R2':<16}{'MAE':<16}\n" + "-" * 78)
    for m in results:
        for s in splits:
            t = results[m][s]
            rmse = np.array([x["test_metrics"]["rmse"] for x in t])
            r2 = np.array([x["test_metrics"]["r2"] for x in t])
            mae = np.array([x["test_metrics"]["mae"] for x in t])
            print(f"{m:<16}{s:<8}{rmse.mean():>7.3f}+/-{rmse.std():<6.3f} "
                  f"{r2.mean():>6.3f}+/-{r2.std():<5.3f} {mae.mean():>6.3f}+/-{mae.std():<5.3f}")
    print("-" * 78 + "\nNaive baselines (RMSE by split):")
    for bn in ["last_value", "moving_avg_4w", "croston"]:
        print("  " + f"{bn:<14}" + "".join(f"{baselines[bn][s]['rmse']:>8.3f}  " for s in splits))
    print("\n[sku_sales] EXPERIMENT 3 COMPLETE.")


if __name__ == "__main__":
    main()
