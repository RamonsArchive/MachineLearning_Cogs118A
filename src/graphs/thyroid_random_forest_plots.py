import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def plot_thyroid_random_forest_summary(results_rf, save_dir):
    """
    Plot summary for Random Forest on the Thyroid Cancer dataset.
    """

    os.makedirs(save_dir, exist_ok=True)

    split_names = []
    mean_train = []
    mean_val = []
    mean_test = []

    best_model = None

    for split_name, trials in results_rf.items():
        split_names.append(split_name)

        train_scores = [t["cv_train_score"] for t in trials]
        val_scores = [t["cv_val_score"] for t in trials]
        test_accs = [t["test_accuracy"] for t in trials]

        mean_train.append(np.mean(train_scores))
        mean_val.append(np.mean(val_scores))
        mean_test.append(np.mean(test_accs))

        for idx, t in enumerate(trials):
            if best_model is None or t["test_metrics"]["roc_auc"] > best_model["record"]["test_metrics"]["roc_auc"]:
                best_model = {
                    "split_name": split_name,
                    "trial_index": idx,
                    "record": t,
                }

    # Accuracy vs split
    plt.figure(figsize=(8, 5))
    plt.plot(split_names, mean_train, marker="o", linewidth=2, markersize=8, label="Train (CV mean)")
    plt.plot(split_names, mean_val, marker="s", linewidth=2, markersize=8, label="Validation (CV mean)")
    plt.plot(split_names, mean_test, marker="^", linewidth=2, markersize=8, label="Test (mean of trials)")
    plt.xlabel("Train/Test Split", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Thyroid Cancer – Random Forest: Performance vs Train/Test Split", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "thyroid_rf_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curve for best model (if probabilities available)
    roc_auc_val = None
    if best_model is not None and best_model["record"]["y_proba"] is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_proba = np.array(rec["y_proba"])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
        plt.fill_between(fpr, tpr, alpha=0.2, color="#2563eb")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title("Thyroid Cancer – Random Forest ROC Curve (Best Model)", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "thyroid_rf_roc_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Confusion matrix & feature importances for best model
    if best_model is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_pred = np.array(rec["y_pred"])

        cm = confusion_matrix(y_test, y_pred)
        classes = ["No", "Yes"]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Thyroid Cancer – Random Forest Confusion Matrix (best model)", fontsize=12)
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
        plt.savefig(os.path.join(save_dir, "thyroid_rf_confusion_best_model.png"), dpi=150, bbox_inches="tight")
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
            plt.title("Thyroid Cancer – Random Forest Top 10 Feature Importances", fontsize=12)
            plt.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "thyroid_rf_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Text report
    report_path = os.path.join(save_dir, "thyroid_random_forest_report.txt")
    with open(report_path, "w") as f:
        f.write("Thyroid Cancer Dataset – Random Forest Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("PERFORMANCE BY SPLIT (averaged over 3 trials)\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            f.write(
                f"{split_name:<10} "
                f"Train (CV) = {mean_train[i]:.4f}, "
                f"Val (CV) = {mean_val[i]:.4f}, "
                f"Test Acc = {mean_test[i]:.4f}\n"
            )
        f.write("\n")

        if best_model is not None:
            split_name = best_model["split_name"]
            idx = best_model["trial_index"]
            rec = best_model["record"]

            f.write("Best model (by ROC-AUC):\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Split: {split_name}\n")
            f.write(f"  Trial index: {idx}\n")
            f.write(f"  Best params: {rec.get('best_params', {})}\n\n")

            tm = rec["test_metrics"]
            f.write("  Test metrics:\n")
            for k, v in tm.items():
                f.write(f"    {k}: {v:.4f}\n")

            if roc_auc_val is not None:
                f.write(f"\n  ROC-AUC used in ROC plot: {roc_auc_val:.4f}\n")

            y_test = np.array(rec["y_test"])
            y_pred = np.array(rec["y_pred"])
            f.write("\nClassification report (test set):\n")
            f.write(classification_report(y_test, y_pred))
        else:
            f.write("No best model found (no trials?)\n")

    print(f"[thyroid_random_forest_plots] Saved plots and report to {save_dir}")


