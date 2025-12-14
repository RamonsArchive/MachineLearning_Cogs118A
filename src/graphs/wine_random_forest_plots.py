import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix, classification_report


def plot_wine_random_forest_summary(results_rf, save_dir):
    """
    Plot summary for Random Forest on the Wine dataset.

    results_rf: results["random_forest"] from wine.py
    """

    os.makedirs(save_dir, exist_ok=True)

    split_names = []
    mean_train = []
    mean_val = []
    mean_test_acc = []
    mean_test_f1 = []
    mean_test_roc_auc = []

    best_model = None

    for split_name, trials in results_rf.items():
        split_names.append(split_name)

        train_scores = [t["cv_train_score"] for t in trials]
        val_scores = [t["cv_val_score"] for t in trials]
        test_accs = [t["test_metrics"]["accuracy"] for t in trials]
        test_f1s = [t["test_metrics"]["f1"] for t in trials]
        test_roc_aucs = [t["test_metrics"].get("roc_auc", 0) for t in trials]

        mean_train.append(np.mean(train_scores))
        mean_val.append(np.mean(val_scores))
        mean_test_acc.append(np.mean(test_accs))
        mean_test_f1.append(np.mean(test_f1s))
        mean_test_roc_auc.append(np.mean(test_roc_aucs) if any(test_roc_aucs) else 0)

        for idx, t in enumerate(trials):
            if best_model is None or t["test_accuracy"] > best_model["record"]["test_accuracy"]:
                best_model = {
                    "split_name": split_name,
                    "trial_index": idx,
                    "record": t,
                }

    # Accuracy vs split
    plt.figure(figsize=(8, 5))
    plt.plot(split_names, mean_train, marker="o", linewidth=2, markersize=8, label="Train (CV mean)")
    plt.plot(split_names, mean_val, marker="s", linewidth=2, markersize=8, label="Validation (CV mean)")
    plt.plot(split_names, mean_test_acc, marker="^", linewidth=2, markersize=8, label="Test (mean of trials)")
    plt.xlabel("Train/Test Split", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Wine Dataset – Random Forest: Accuracy vs Train/Test Split", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wine_rf_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix & feature importance for best model
    if best_model is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_pred = np.array(rec["y_pred"])

        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Greens")
        plt.title("Wine Dataset – Random Forest Confusion Matrix (best model)", fontsize=12)
        plt.colorbar(im)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        plt.xlabel("Predicted label", fontsize=11)
        plt.ylabel("True label", fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "wine_rf_confusion_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Feature importances (top 10)
        feat_importances = rec.get("feature_importances")
        feat_names = rec.get("feature_names")
        if feat_importances is not None and feat_names is not None:
            feat_importances = np.array(feat_importances)
            sorted_idx = np.argsort(feat_importances)[::-1][:10]
            top_importances = feat_importances[sorted_idx]
            top_names = [feat_names[i] for i in sorted_idx]

            plt.figure(figsize=(8, 5))
            y_pos = np.arange(len(top_names))
            plt.barh(y_pos, top_importances[::-1], color="#2563eb", alpha=0.8)
            plt.yticks(y_pos, top_names[::-1], fontsize=9)
            plt.xlabel("Feature Importance", fontsize=11)
            plt.title("Wine Dataset – Random Forest Top 10 Feature Importances", fontsize=12)
            plt.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "wine_rf_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Text report
    report_path = os.path.join(save_dir, "wine_random_forest_report.txt")
    with open(report_path, "w") as f:
        f.write("Wine Dataset – Random Forest Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("PERFORMANCE BY SPLIT (averaged over 3 trials)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Split':<10} {'CV Train':<12} {'CV Val':<12} {'Test Acc':<12} {'Test F1':<12} {'Test ROC-AUC':<12}\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            roc_auc_str = f"{mean_test_roc_auc[i]:.4f}" if mean_test_roc_auc[i] > 0 else "N/A"
            f.write(
                f"{split_name:<10} "
                f"{mean_train[i]:<12.4f} "
                f"{mean_val[i]:<12.4f} "
                f"{mean_test_acc[i]:<12.4f} "
                f"{mean_test_f1[i]:<12.4f} "
                f"{roc_auc_str:<12}\n"
            )
        f.write("\n")

        if best_model is not None:
            split_name = best_model["split_name"]
            idx = best_model["trial_index"]
            rec = best_model["record"]

            f.write("Best model (by test accuracy):\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Split: {split_name}\n")
            f.write(f"  Trial index: {idx}\n")
            f.write(f"  Best params: {rec.get('best_params', {})}\n\n")

            tm = rec["test_metrics"]
            f.write("  Test metrics:\n")
            for k, v in tm.items():
                f.write(f"    {k}: {v:.4f}\n")

            y_test = np.array(rec["y_test"])
            y_pred = np.array(rec["y_pred"])
            f.write("\nClassification report (test set):\n")
            f.write(classification_report(y_test, y_pred))
        else:
            f.write("No best model found (no trials?)\n")

    print(f"[wine_random_forest_plots] Saved plots and report to {save_dir}")


