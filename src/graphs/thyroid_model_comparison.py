import os
import numpy as np
import matplotlib.pyplot as plt


def plot_thyroid_model_comparison(all_results, save_dir):
    """
    Compare boosting, random_forest, neural_network on Thyroid Cancer (binary).
    Primary metric: test ROC-AUC (falls back to accuracy if ROC-AUC is NaN).
    Expects structure:
      all_results = {
        "boosting": { split: [trial_records...] },
        "random_forest": { ... },
        "neural_network": { ... },
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    splits = list(next(iter(all_results.values())).keys())

    def metric_from_trial(t):
        roc = t["test_metrics"].get("roc_auc", np.nan)
        if np.isnan(roc):
            return t["test_metrics"].get("accuracy", np.nan)
        return roc

    # Bar chart of mean ROC-AUC (or accuracy) per model per split
    x = np.arange(len(splits))
    width = 0.25
    colors = ["#2563eb", "#10b981", "#7c3aed"]

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        mean_scores = []
        for split in splits:
            scores = [metric_from_trial(t) for t in all_results[model][split]]
            mean_scores.append(np.nanmean(scores))
        plt.bar(x + (i - 1) * width, mean_scores, width, label=model.replace("_", " ").title(), color=colors[i % len(colors)])

    plt.xlabel("Train/Test Split", fontsize=12)
    plt.ylabel("Test ROC-AUC (fallback: Accuracy)", fontsize=12)
    plt.title("Thyroid Cancer – Model Comparison", fontsize=14)
    plt.xticks(x, splits)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "thyroid_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Text report
    report_path = os.path.join(save_dir, "thyroid_model_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("Thyroid Cancer Dataset – Model Comparison Report\n")
        f.write("=" * 70 + "\n\n")

        for split in splits:
            f.write(f"Split {split}:\n")
            for model in model_names:
                scores = [metric_from_trial(t) for t in all_results[model][split]]
                f.write(f"  {model}: mean={np.nanmean(scores):.4f}, std={np.nanstd(scores):.4f}\n")
            f.write("\n")

        best_model = None
        best_score = -np.inf
        for model in model_names:
            scores_all = []
            for split in splits:
                scores_all.extend([metric_from_trial(t) for t in all_results[model][split]])
            mean_score = np.nanmean(scores_all)
            if mean_score > best_score:
                best_score = mean_score
                best_model = model

        f.write("Best overall model by mean ROC-AUC (fallback accuracy):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Model: {best_model}\n")
        f.write(f"  Mean score: {best_score:.4f}\n")

    print(f"[thyroid_model_comparison] Saved comparison plot and report to {save_dir}")


