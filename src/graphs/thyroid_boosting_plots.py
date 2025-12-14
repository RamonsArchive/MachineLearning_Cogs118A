import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def plot_thyroid_boosting_summary(results_boosting, save_dir):
    """
    Plot summary for XGBoost (boosting) on the Thyroid Cancer dataset.

    Binary classification: target 'Recurred' (Yes/No).
    """

    os.makedirs(save_dir, exist_ok=True)

    split_names = []
    mean_train = []
    mean_val = []
    mean_test_acc = []
    mean_test_f1 = []
    mean_test_roc_auc = []

    best_model = None

    for split_name, trials in results_boosting.items():
        split_names.append(split_name)

        train_scores = [t["cv_train_score"] for t in trials]
        val_scores = [t["cv_val_score"] for t in trials]
        test_accs = [t["test_metrics"]["accuracy"] for t in trials]
        test_f1s = [t["test_metrics"]["f1"] for t in trials]
        test_roc_aucs = [t["test_metrics"]["roc_auc"] for t in trials]

        mean_train.append(np.mean(train_scores))
        mean_val.append(np.mean(val_scores))
        mean_test_acc.append(np.mean(test_accs))
        mean_test_f1.append(np.mean(test_f1s))
        mean_test_roc_auc.append(np.mean(test_roc_aucs))

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
    plt.plot(split_names, mean_test_acc, marker="^", linewidth=2, markersize=8, label="Test (mean of trials)")
    plt.xlabel("Train/Test Split", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Thyroid Cancer – XGBoost: Performance vs Train/Test Split", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "thyroid_boosting_accuracy.png"), dpi=150, bbox_inches="tight")
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
        plt.plot(fpr, tpr, color="#10b981", linewidth=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
        plt.fill_between(fpr, tpr, alpha=0.2, color="#10b981")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title("Thyroid Cancer – XGBoost ROC Curve (Best Model)", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "thyroid_boosting_roc_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Confusion matrix for best model
    if best_model is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_pred = np.array(rec["y_pred"])

        cm = confusion_matrix(y_test, y_pred)
        classes = ["No", "Yes"]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Greens")
        plt.title("Thyroid Cancer – XGBoost Confusion Matrix (best model)", fontsize=12)
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
        plt.savefig(os.path.join(save_dir, "thyroid_boosting_confusion_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Text report
    report_path = os.path.join(save_dir, "thyroid_boosting_report.txt")
    with open(report_path, "w") as f:
        f.write("Thyroid Cancer Dataset – XGBoost (Boosting) Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("PERFORMANCE BY SPLIT (averaged over 3 trials)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Split':<10} {'CV Train':<12} {'CV Val':<12} {'Test Acc':<12} {'Test F1':<12} {'Test ROC-AUC':<12}\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            f.write(
                f"{split_name:<10} "
                f"{mean_train[i]:<12.4f} "
                f"{mean_val[i]:<12.4f} "
                f"{mean_test_acc[i]:<12.4f} "
                f"{mean_test_f1[i]:<12.4f} "
                f"{mean_test_roc_auc[i]:<12.4f}\n"
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

    print(f"[thyroid_boosting_plots] Saved plots and report to {save_dir}")


