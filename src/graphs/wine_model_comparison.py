import os
import numpy as np
import matplotlib.pyplot as plt


def plot_wine_model_comparison(all_results, save_dir):
    """
    Compare boosting, random_forest, neural_network on Wine (multiclass).
    Uses test accuracy as the primary metric.
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

    # Build bar chart of mean test F1 score per model per split
    x = np.arange(len(splits))
    width = 0.25
    colors = ["#2563eb", "#10b981", "#7c3aed"]

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        mean_f1 = []
        for split in splits:
            f1s = [t["test_metrics"]["f1"] for t in all_results[model][split]]
            mean_f1.append(np.mean(f1s))
        plt.bar(x + (i - 1) * width, mean_f1, width, label=model.replace("_", " ").title(), color=colors[i % len(colors)])

    plt.xlabel("Train/Test Split", fontsize=12)
    plt.ylabel("Test F1 Score", fontsize=12)
    plt.title("Wine Dataset – Model Comparison (F1 Score)", fontsize=14)
    plt.xticks(x, splits)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wine_model_comparison_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Text report
    report_path = os.path.join(save_dir, "wine_model_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("Wine Dataset – Model Comparison Report\n")
        f.write("=" * 70 + "\n\n")

        for split in splits:
            f.write(f"Split {split}:\n")
            for model in model_names:
                f1s = [t["test_metrics"]["f1"] for t in all_results[model][split]]
                f.write(f"  {model}: mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}\n")
            f.write("\n")

        # Best model overall by mean F1 score across splits
        best_model = None
        best_score = -np.inf
        for model in model_names:
            f1s_all = []
            for split in splits:
                f1s_all.extend([t["test_metrics"]["f1"] for t in all_results[model][split]])
            mean_f1 = np.mean(f1s_all)
            if mean_f1 > best_score:
                best_score = mean_f1
                best_model = model

        f.write("Best overall model by mean test F1 score:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Model: {best_model}\n")
        f.write(f"  Mean F1: {best_score:.4f}\n")

    print(f"[wine_model_comparison] Saved comparison plot and report to {save_dir}")


