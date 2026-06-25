"""
Plots + text reports for the Shopify SKU-sales regression (Experiment 3).

Same conventions as graphs/forecast_demand_plots.py (per-model *_report.txt with
mean +/- std over trials, RMSE primary, MAE on non-zero weeks, naive-baseline floor,
self-tagging split x trial header, scatter/residual/importance/by-split plots, and a
comparison/ report) - a dedicated module per harness convention (bank_*, wine_*, ...).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

PRETTY = {"boosting": "XGBoost", "random_forest": "Random Forest",
          "neural_network": "Neural Network"}
LABEL = "MODALAI SHOPIFY SKU SALES"
PREFIX = "sku_sales"


def _ms(v): return float(np.mean(v)), float(np.std(v))


def _report(results, model_name, output_dir):
    pretty = PRETTY.get(model_name, model_name)
    nsp = len(results); ntr = len(next(iter(results.values()))) if results else 0
    kind = "SMOKE TEST" if (nsp <= 1 and ntr <= 1) else "full run"
    L = ["=" * 78, f"{LABEL} - {pretty.upper()} RESULTS", "=" * 78,
         "Target: units_sold (winsorized + log1p; metrics inverted to real units)",
         f"Primary metric: RMSE (lower=better). Temporal split; {nsp} split(s) x "
         f"{ntr} trial(s)  [{kind}].",
         "\n" + "-" * 70,
         f"PERFORMANCE BY SPLIT (mean +/- std over {ntr} trial(s))", "-" * 70,
         f"{'Split':<8}{'Test RMSE':<20}{'Test MAE':<18}{'MAE!=0':<18}{'Test R2':<16}",
         "-" * 70]
    for s, trials in results.items():
        rm = _ms([t["test_metrics"]["rmse"] for t in trials])
        ma = _ms([t["test_metrics"]["mae"] for t in trials])
        nz = _ms([t["test_metrics"]["mae_nonzero"] for t in trials])
        r2 = _ms([t["test_metrics"]["r2"] for t in trials])
        L.append(f"{s:<8}{rm[0]:>8.3f} +/- {rm[1]:<6.3f} {ma[0]:>7.3f} +/- {ma[1]:<5.3f} "
                 f"{nz[0]:>7.2f} +/- {nz[1]:<5.2f} {r2[0]:>6.3f} +/- {r2[1]:<5.3f}")
    best, bsplit, brmse = None, None, float("inf")
    for s, trials in results.items():
        for t in trials:
            if t["test_metrics"]["rmse"] < brmse:
                brmse, best, bsplit = t["test_metrics"]["rmse"], t, s
    if best:
        L += ["\n" + "-" * 70, "BEST MODEL (lowest test RMSE)", "-" * 70,
              f"Split: {bsplit}   Trial: {best['trial']+1}", "\nBest hyperparameters:"]
        for k, v in best["best_params"].items():
            L.append(f"  {k}: {v}")
        m = best["test_metrics"]
        L += ["\nTest metrics (real units):", f"  RMSE:      {m['rmse']:.4f}",
              f"  MAE:       {m['mae']:.4f}", f"  MAE (!=0): {m['mae_nonzero']:.4f}",
              f"  R2:        {m['r2']:.4f}"]
    with open(os.path.join(output_dir, f"{PREFIX}_{model_name}_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    return best


def _scatter(best, model_name, output_dir):
    yt = np.log1p(np.array(best["y_test"], float)); yp = np.log1p(np.array(best["y_pred"], float))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt, yp, alpha=0.25, s=10, c="teal")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
    ax.set_xlabel("log1p(actual units)"); ax.set_ylabel("log1p(predicted units)")
    ax.set_title(f"{PRETTY.get(model_name, model_name)}: Predicted vs Actual\n"
                 f"R2={best['test_metrics']['r2']:.4f}, RMSE={best['test_metrics']['rmse']:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{PREFIX}_{model_name}_scatter.png"), dpi=150)
    plt.close()


def _importance(best, model_name, output_dir):
    if not best.get("feature_importances") or not best.get("feature_names"):
        return
    imp = np.array(best["feature_importances"], float); names = best["feature_names"]
    idx = np.argsort(imp)[-15:][::-1]
    fig, ax = plt.subplots(figsize=(10, 8)); yp = np.arange(len(idx))
    ax.barh(yp, imp[idx], color="teal", alpha=0.85); ax.set_yticks(yp)
    ax.set_yticklabels([names[i] for i in idx]); ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top 15 Feature Importances ({PRETTY.get(model_name, model_name)})")
    ax.grid(True, axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{PREFIX}_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


def _rmse_by_split(results, model_name, output_dir):
    splits = list(results.keys())
    means, stds, pts = [], [], []
    for s in splits:
        r = [t["test_metrics"]["rmse"] for t in results[s]]
        m, sd = _ms(r); means.append(m); stds.append(sd); pts.append(r)
    x = np.arange(len(splits)); fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color="teal", alpha=0.8)
    for i, r in enumerate(pts):
        ax.scatter([x[i]] * len(r), r, color="black", zorder=3, s=25)
    ax.set_xlabel("Temporal split"); ax.set_ylabel("Test RMSE (units)")
    ax.set_title(f"{PRETTY.get(model_name, model_name)}: Test RMSE by split")
    ax.set_xticks(x); ax.set_xticklabels(splits); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{PREFIX}_{model_name}_rmse_by_split.png"), dpi=150)
    plt.close()


def _summary(results, model_name, output_dir, importance=False):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[plot] {model_name} -> {output_dir}")
    best = _report(results, model_name, output_dir)
    if best:
        _scatter(best, model_name, output_dir)
        if importance:
            _importance(best, model_name, output_dir)
    _rmse_by_split(results, model_name, output_dir)


def plot_sku_sales_boosting_summary(results, d): _summary(results, "boosting", d, True)
def plot_sku_sales_random_forest_summary(results, d): _summary(results, "random_forest", d, True)
def plot_sku_sales_neural_network_summary(results, d): _summary(results, "neural_network", d, False)


def plot_sku_sales_model_comparison(all_results, baselines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = list(all_results.keys()); splits = list(all_results[models[0]].keys())
    colors = ["teal", "darkorange", "purple"]
    x = np.arange(len(splits)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mdl in enumerate(models):
        means = [np.mean([t["test_metrics"]["rmse"] for t in all_results[mdl][s]]) for s in splits]
        ax.bar(x + (i - (len(models)-1)/2)*w, means, w, label=PRETTY.get(mdl, mdl),
               color=colors[i % len(colors)], alpha=0.85)
    for j, (bn, bs) in enumerate(baselines.items()):
        ax.axhline(np.mean([bs[s]["rmse"] for s in splits]), ls="--", lw=1.3, alpha=0.7,
                   color=["black", "dimgray", "darkred"][j % 3], label=f"baseline: {bn}")
    ax.set_xlabel("Temporal split"); ax.set_ylabel("Test RMSE (lower=better)")
    ax.set_title(f"{LABEL}: RMSE by split (models vs baselines)")
    ax.set_xticks(x); ax.set_xticklabels(splits); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_rmse.png"), dpi=150); plt.close()

    nsp = len(splits); ntr = len(all_results[models[0]][splits[0]])
    kind = "SMOKE TEST" if (nsp <= 1 and ntr <= 1) else "full run"
    hdr = f"{'Model':<18}" + "".join(f"{s:<14}" for s in splits) + f"{'Overall':<12}"
    L = ["=" * 80, f"{LABEL} - MODEL COMPARISON SUMMARY", "=" * 80,
         f"Primary metric: RMSE (real units). Temporal split; {nsp} split(s) x "
         f"{ntr} trial(s) [{kind}].", "\n" + "-" * 72,
         "AVERAGE TEST RMSE BY MODEL AND SPLIT", "-" * 72, hdr, "-" * 72]
    for mdl in models:
        vals = [np.mean([t["test_metrics"]["rmse"] for t in all_results[mdl][s]]) for s in splits]
        L.append(f"{PRETTY.get(mdl, mdl):<18}" + "".join(f"{v:<14.3f}" for v in vals)
                 + f"{np.mean(vals):<12.3f}")
    L += ["." * 72, "NAIVE BASELINES (the floor models must beat):"]
    for bn, bs in baselines.items():
        vals = [bs[s]["rmse"] for s in splits]
        L.append(f"{bn:<18}" + "".join(f"{v:<14.3f}" for v in vals) + f"{np.mean(vals):<12.3f}")
    # best non-degenerate
    bm, br, bsp, brec = None, float("inf"), None, None
    for mdl in models:
        for s in splits:
            for t in all_results[mdl][s]:
                tm = t["test_metrics"]
                if tm["rmse"] < 1e-9 or not np.isfinite(tm.get("mae_nonzero", np.nan)):
                    continue
                if tm["rmse"] < br:
                    br, bm, bsp, brec = tm["rmse"], mdl, s, t
    L += ["\n" + "-" * 72, "BEST OVERALL MODEL (lowest single-trial test RMSE)", "-" * 72]
    if bm:
        L += [f"Model: {PRETTY.get(bm, bm)}", f"Split: {bsp}", f"RMSE:      {br:.4f}",
              f"MAE (!=0): {brec['test_metrics']['mae_nonzero']:.4f}",
              f"R2:        {brec['test_metrics']['r2']:.4f}", "\nvs baselines on that split:"]
        for bn, bs in baselines.items():
            b = bs[bsp]["rmse"]
            L.append(f"  {bn:<14} {b:.3f}  -> model {'BEATS' if br < b else 'loses to'} it")
    with open(os.path.join(output_dir, "model_comparison_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[plot] saved comparison -> {output_dir}")
