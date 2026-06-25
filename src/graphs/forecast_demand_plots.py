"""
Plotting + text-report functions for the ModalAI demand-forecasting regression.

Mirrors graphs/parkinsons_plots.py conventions (per-model *_report.txt, scatter,
residuals, feature importance, metric-by-split bars, plus a comparison/ report) and
extends them for this dataset:
  - PRIMARY metric is RMSE (lower is better) - matches the other regressions.
  - reports show mean +/- std across the 3 trials per split.
  - reports add MAE on NON-ZERO part-weeks (the metric that matters under 94% zeros).
  - the comparison report ranks the tuned models against the naive baselines
    (last-value, 4-week moving-average, Croston) so models are judged vs a fair floor.
  - scatter/residuals are drawn on a log1p axis (raw demand is too heavy-tailed to read).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

PRETTY = {
    "boosting": "XGBoost",
    "random_forest": "Random Forest",
    "neural_network": "Neural Network",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _ms(vals):
    """mean, std tuple."""
    return float(np.mean(vals)), float(np.std(vals))


def _best_trial_lowest_rmse(results):
    best, best_rmse, best_split = None, float("inf"), None
    for split_name, trials in results.items():
        for t in trials:
            if t["test_metrics"]["rmse"] < best_rmse:
                best_rmse, best, best_split = t["test_metrics"]["rmse"], t, split_name
    return best, best_split


# ----------------------------------------------------------------------------
# per-model text report
# ----------------------------------------------------------------------------
def _generate_regression_report(results, model_name, output_dir):
    pretty = PRETTY.get(model_name, model_name)
    n_splits = len(results)
    n_trials = len(next(iter(results.values()))) if results else 0
    run_kind = "SMOKE TEST" if (n_splits <= 1 and n_trials <= 1) else "full run"
    L = []
    L.append("=" * 78)
    L.append(f"MODALAI DEMAND FORECAST - {pretty.upper()} RESULTS")
    L.append("=" * 78)
    L.append("Target: demand_qty (winsorized + log1p; metrics inverted to real units)")
    L.append(f"Primary metric: RMSE (units, lower is better). Temporal split; "
             f"{n_splits} split(s) x {n_trials} trial(s)  [{run_kind}].")

    L.append("\n" + "-" * 70)
    L.append(f"PERFORMANCE BY SPLIT  (mean +/- std over {n_trials} trial(s))")
    L.append("-" * 70)
    L.append(f"{'Split':<8}{'Test RMSE':<20}{'Test MAE':<18}"
             f"{'MAE!=0':<18}{'Test R2':<16}")
    L.append("-" * 70)
    for split_name, trials in results.items():
        rmse_m, rmse_s = _ms([t["test_metrics"]["rmse"] for t in trials])
        mae_m, mae_s = _ms([t["test_metrics"]["mae"] for t in trials])
        nz_m, nz_s = _ms([t["test_metrics"]["mae_nonzero"] for t in trials])
        r2_m, r2_s = _ms([t["test_metrics"]["r2"] for t in trials])
        L.append(f"{split_name:<8}"
                 f"{rmse_m:>8.3f} +/- {rmse_s:<6.3f} "
                 f"{mae_m:>7.3f} +/- {mae_s:<5.3f} "
                 f"{nz_m:>7.2f} +/- {nz_s:<5.2f} "
                 f"{r2_m:>6.3f} +/- {r2_s:<5.3f}")

    L.append("\n" + "-" * 70)
    L.append("CV (on log target, neg_MSE) + OVERFITTING CHECK  (mean over trials)")
    L.append("-" * 70)
    L.append(f"{'Split':<8}{'CV train':<14}{'CV val':<14}"
             f"{'Train RMSE':<14}{'Test RMSE':<14}")
    L.append("-" * 70)
    for split_name, trials in results.items():
        cvt, _ = _ms([t["cv_train_score"] for t in trials])
        cvv, _ = _ms([t["cv_val_score"] for t in trials])
        trrmse, _ = _ms([t["train_metrics"]["rmse"] for t in trials])
        termse, _ = _ms([t["test_metrics"]["rmse"] for t in trials])
        L.append(f"{split_name:<8}{cvt:<14.4f}{cvv:<14.4f}{trrmse:<14.3f}{termse:<14.3f}")

    best, best_split = _best_trial_lowest_rmse(results)
    if best:
        L.append("\n" + "-" * 70)
        L.append("BEST MODEL (lowest test RMSE)")
        L.append("-" * 70)
        L.append(f"Split: {best_split}   Trial: {best['trial'] + 1}")
        L.append("\nBest hyperparameters:")
        for k, v in best["best_params"].items():
            L.append(f"  {k}: {v}")
        m = best["test_metrics"]
        L.append("\nTest metrics (real units):")
        L.append(f"  RMSE:        {m['rmse']:.4f}")
        L.append(f"  MAE:         {m['mae']:.4f}")
        L.append(f"  MAE (!=0):   {m['mae_nonzero']:.4f}")
        L.append(f"  R2:          {m['r2']:.4f}")
        L.append(f"  MSE:         {m['mse']:.4f}")

    L.append("\n" + "-" * 70)
    L.append("INTERPRETATION")
    L.append("-" * 70)
    if best:
        r2 = best["test_metrics"]["r2"]
        quality = ("EXCELLENT" if r2 > 0.9 else "GOOD" if r2 > 0.7
                   else "MODERATE" if r2 > 0.5 else "WEAK" if r2 > 0.2 else "POOR")
        L.append(f"Model quality: {quality} (best test R2 = {r2:.4f})")
        L.append(f"Explains {r2*100:.1f}% of variance in weekly demand on held-out weeks.")
        L.append(f"Typical error on weeks that DO have demand: "
                 f"MAE!=0 ~ {best['test_metrics']['mae_nonzero']:.2f} units.")

    report_path = os.path.join(output_dir, f"forecast_demand_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return best


# ----------------------------------------------------------------------------
# per-model figures
# ----------------------------------------------------------------------------
def _plot_pred_scatter(best, model_name, output_dir):
    pretty = PRETTY.get(model_name, model_name)
    yt = np.log1p(np.array(best["y_test"], dtype=float))
    yp = np.log1p(np.array(best["y_pred"], dtype=float))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt, yp, alpha=0.25, s=10, c="steelblue")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
    ax.set_xlabel("log1p(actual demand)")
    ax.set_ylabel("log1p(predicted demand)")
    ax.set_title(f"{pretty}: Predicted vs Actual\n"
                 f"R2 = {best['test_metrics']['r2']:.4f}, "
                 f"RMSE = {best['test_metrics']['rmse']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forecast_demand_{model_name}_scatter.png"),
                dpi=150)
    plt.close()


def _plot_residuals(best, model_name, output_dir):
    pretty = PRETTY.get(model_name, model_name)
    yt = np.array(best["y_test"], dtype=float)
    yp = np.array(best["y_pred"], dtype=float)
    resid = np.log1p(yt) - np.log1p(yp)  # residuals on log scale (readable)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(resid, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--", lw=2, label="Zero error")
    axes[0].axvline(resid.mean(), color="orange", linestyle="--", lw=2,
                    label=f"Mean: {resid.mean():.3f}")
    axes[0].set_xlabel("Residual log1p(actual) - log1p(pred)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual distribution (log scale)")
    axes[0].legend()
    axes[1].scatter(np.log1p(yp), resid, alpha=0.25, s=10, c="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("log1p(predicted demand)")
    axes[1].set_ylabel("Residual (log scale)")
    axes[1].set_title(f"{pretty}: Residuals vs Predicted")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forecast_demand_{model_name}_residuals.png"),
                dpi=150)
    plt.close()


def _plot_feature_importance(best, model_name, output_dir):
    if not best.get("feature_importances") or not best.get("feature_names"):
        return
    pretty = PRETTY.get(model_name, model_name)
    imp = np.array(best["feature_importances"], dtype=float)
    names = best["feature_names"]
    idx = np.argsort(imp)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ypos = np.arange(len(idx))
    ax.barh(ypos, imp[idx], color="steelblue", alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top 15 Feature Importances ({pretty})")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"forecast_demand_{model_name}_feature_importance.png"),
        dpi=150)
    plt.close()


def _plot_rmse_by_split(results, model_name, output_dir):
    """Primary-metric (RMSE) bars + a per-trial overlay (the per-trial view)."""
    pretty = PRETTY.get(model_name, model_name)
    splits = list(results.keys())
    means, stds, per_trial = [], [], []
    for s in splits:
        rmses = [t["test_metrics"]["rmse"] for t in results[s]]
        m, sd = _ms(rmses)
        means.append(m); stds.append(sd); per_trial.append(rmses)
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    for i, rmses in enumerate(per_trial):  # per-trial points overlaid
        ax.scatter([x[i]] * len(rmses), rmses, color="black", zorder=3, s=25)
    ax.set_xlabel("Temporal train/test split")
    ax.set_ylabel("Test RMSE (units)")
    ax.set_title(f"{pretty}: Test RMSE by split (bars=mean+/-std, dots=trials)")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forecast_demand_{model_name}_rmse_by_split.png"),
                dpi=150)
    plt.close()


def _summary(results, model_name, output_dir, feature_importance=False):
    os.makedirs(output_dir, exist_ok=True)
    pretty = PRETTY.get(model_name, model_name)
    print(f"[plot] Generating {model_name} ({pretty}) plots to {output_dir}")
    best = _generate_regression_report(results, model_name, output_dir)
    if best:
        _plot_pred_scatter(best, model_name, output_dir)
        _plot_residuals(best, model_name, output_dir)
        if feature_importance:
            _plot_feature_importance(best, model_name, output_dir)
    _plot_rmse_by_split(results, model_name, output_dir)
    print(f"[plot] Saved {model_name} plots and report")


# public per-model entry points (named like the parkinsons module)
def plot_forecast_demand_boosting_summary(results, output_dir):
    _summary(results, "boosting", output_dir, feature_importance=True)


def plot_forecast_demand_random_forest_summary(results, output_dir):
    _summary(results, "random_forest", output_dir, feature_importance=True)


def plot_forecast_demand_neural_network_summary(results, output_dir):
    _summary(results, "neural_network", output_dir, feature_importance=False)


# ----------------------------------------------------------------------------
# comparison (models + naive baselines)
# ----------------------------------------------------------------------------
def plot_forecast_demand_model_comparison(all_results, baselines, output_dir):
    """
    all_results: {model_name: {split: [trial_records]}}
    baselines:   {baseline_name: {split: {"rmse","mae","mae_nonzero","r2"}}}
                 (deterministic per split -> single dict, no trials)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[plot] Generating model comparison to {output_dir}")

    model_names = list(all_results.keys())
    splits = list(all_results[model_names[0]].keys())
    colors = ["steelblue", "forestgreen", "coral", "purple"]

    # ---- RMSE grouped bars (models only) ----
    x = np.arange(len(splits))
    width = 0.8 / max(len(model_names), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(model_names):
        means = [np.mean([t["test_metrics"]["rmse"] for t in all_results[model][s]])
                 for s in splits]
        stds = [np.std([t["test_metrics"]["rmse"] for t in all_results[model][s]])
                for s in splits]
        ax.bar(x + (i - (len(model_names) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=PRETTY.get(model, model),
               color=colors[i % len(colors)], alpha=0.85)
    # baseline reference lines (mean RMSE across splits per baseline)
    for j, (bname, bsplits) in enumerate(baselines.items()):
        avg = np.mean([bsplits[s]["rmse"] for s in splits])
        ax.axhline(avg, linestyle="--", lw=1.5, alpha=0.7,
                   color=["black", "dimgray", "darkred"][j % 3],
                   label=f"baseline: {bname} (avg {avg:.1f})")
    ax.set_xlabel("Temporal split")
    ax.set_ylabel("Test RMSE (units, lower is better)")
    ax.set_title("Model vs Baseline: RMSE by split (+/- std across trials)")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_rmse.png"), dpi=150)
    plt.close()

    # ---- R2 grouped bars (models only) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(model_names):
        means = [np.mean([t["test_metrics"]["r2"] for t in all_results[model][s]])
                 for s in splits]
        stds = [np.std([t["test_metrics"]["r2"] for t in all_results[model][s]])
                for s in splits]
        ax.bar(x + (i - (len(model_names) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=PRETTY.get(model, model),
               color=colors[i % len(colors)], alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Temporal split")
    ax.set_ylabel("Test R2 (higher is better)")
    ax.set_title("Model Comparison: R2 by split (+/- std across trials)")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_r2.png"), dpi=150)
    plt.close()

    # ---- comparison report ----
    L = []
    n_splits = len(splits)
    n_trials = len(all_results[model_names[0]][splits[0]]) if splits else 0
    run_kind = "SMOKE TEST" if (n_splits <= 1 and n_trials <= 1) else "full run"
    L.append("=" * 80)
    L.append("MODALAI DEMAND FORECAST - MODEL COMPARISON SUMMARY")
    L.append("=" * 80)
    L.append("Primary metric: RMSE (real units, lower is better). Metrics inverted from")
    L.append(f"the winsorized log1p target. Temporal split; {n_splits} split(s) x "
             f"{n_trials} trial(s)  [{run_kind}].")

    L.append("\n" + "-" * 72)
    L.append("AVERAGE TEST RMSE BY MODEL AND SPLIT")
    L.append("-" * 72)
    header = f"{'Model':<18}" + "".join(f"{s:<14}" for s in splits) + f"{'Overall':<12}"
    L.append(header)
    L.append("-" * 72)
    overall = {}
    for model in model_names:
        vals = [np.mean([t["test_metrics"]["rmse"] for t in all_results[model][s]])
                for s in splits]
        overall[model] = np.mean(vals)
        L.append(f"{PRETTY.get(model, model):<18}"
                 + "".join(f"{v:<14.3f}" for v in vals)
                 + f"{overall[model]:<12.3f}")
    L.append("." * 72)
    L.append("NAIVE BASELINES (deterministic per split - the floor models must beat):")
    for bname, bsplits in baselines.items():
        vals = [bsplits[s]["rmse"] for s in splits]
        L.append(f"{bname:<18}" + "".join(f"{v:<14.3f}" for v in vals)
                 + f"{np.mean(vals):<12.3f}")

    # ---- non-zero MAE table (the metric that matters under zero-inflation) ----
    L.append("\n" + "-" * 72)
    L.append("AVERAGE TEST MAE ON NON-ZERO PART-WEEKS  (MAE!=0)")
    L.append("-" * 72)
    L.append(header)
    L.append("-" * 72)
    for model in model_names:
        vals = [np.mean([t["test_metrics"]["mae_nonzero"] for t in all_results[model][s]])
                for s in splits]
        L.append(f"{PRETTY.get(model, model):<18}"
                 + "".join(f"{v:<14.3f}" for v in vals) + f"{np.mean(vals):<12.3f}")
    L.append("." * 72)
    for bname, bsplits in baselines.items():
        vals = [bsplits[s]["mae_nonzero"] for s in splits]
        L.append(f"{bname:<18}" + "".join(f"{v:<14.3f}" for v in vals)
                 + f"{np.mean(vals):<12.3f}")

    # ---- best overall (by single-trial lowest test RMSE) ----
    L.append("\n" + "-" * 72)
    L.append("BEST OVERALL MODEL (lowest single-trial test RMSE)")
    L.append("-" * 72)
    # FIX #3: ignore degenerate trials. A zero-variance test target (e.g. an all-zero
    # demand window) yields RMSE~0 / R2~1.0 with no non-zero weeks (mae_nonzero=nan) -
    # a fake "perfect" score, not a real win. Splits with no train signal are already
    # skipped upstream; this is a belt-and-suspenders guard so such a trial can never be
    # crowned "best overall".
    def _degenerate(tm):
        return tm["rmse"] < 1e-9 or not np.isfinite(tm.get("mae_nonzero", np.nan))

    best_model, best_rmse, best_split, best_rec = None, float("inf"), None, None
    for model in model_names:
        for s in splits:
            for t in all_results[model][s]:
                tm = t["test_metrics"]
                if _degenerate(tm):
                    continue
                if tm["rmse"] < best_rmse:
                    best_rmse = tm["rmse"]
                    best_model, best_split, best_rec = model, s, t
    if best_model:
        L.append(f"Model: {PRETTY.get(best_model, best_model)}")
        L.append(f"Split: {best_split}")
        L.append(f"RMSE:      {best_rmse:.4f}")
        L.append(f"MAE:       {best_rec['test_metrics']['mae']:.4f}")
        L.append(f"MAE (!=0): {best_rec['test_metrics']['mae_nonzero']:.4f}")
        L.append(f"R2:        {best_rec['test_metrics']['r2']:.4f}")
        # did it beat the baselines on that split?
        L.append("\nvs naive baselines on that split (RMSE):")
        for bname, bsplits in baselines.items():
            b = bsplits[best_split]["rmse"]
            verdict = "BEATS" if best_rmse < b else "loses to"
            L.append(f"  {bname:<14} {b:.3f}   -> model {verdict} it")

    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[plot] Saved model comparison plots and report")
