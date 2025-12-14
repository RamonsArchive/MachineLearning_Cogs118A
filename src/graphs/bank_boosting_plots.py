# src/graphs/bank_boosting_plots.py
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def plot_bank_boosting_summary(results_boosting, save_dir):
    """
    results_boosting: results["boosting"] sub-dict from bank.py
    Structure:
    {
        "20_80": [ {trial_record}, {trial_record}, {trial_record} ],
        "50_50": [...],
        "80_20": [...]
    }

    Each trial_record is expected to contain:
      - "cv_train_score": float
      - "cv_val_score": float
      - "test_accuracy": float
      - "test_metrics": dict (with at least "accuracy", "roc_auc", "precision", "recall", "f1")
      - "y_test": list of ints
      - "y_pred": list of ints
      - "y_proba": list of floats or None
    """

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. ROC-AUC vs split (mean over 3 trials) - ALL metrics are ROC-AUC
    # ------------------------------------------------------------------
    split_names = []
    mean_train = []
    mean_val = []
    mean_test_acc = []
    mean_test_f1 = []
    mean_test_roc_auc = []

    # Also keep overall-best model for ROC/confusion matrix
    best_model = None  # dict: {"split_name": ..., "trial_index": ..., "record": ...}

    for split_name, trials in results_boosting.items():
        split_names.append(split_name)

        train_scores = [t["cv_train_score"] for t in trials]  # ROC-AUC
        val_scores = [t["cv_val_score"] for t in trials]      # ROC-AUC
        test_roc_aucs = [t["test_metrics"]["roc_auc"] for t in trials]  # ROC-AUC (consistent!)
        test_accs = [t["test_metrics"]["accuracy"] for t in trials]
        test_f1s = [t["test_metrics"]["f1"] for t in trials]

        mean_train.append(np.mean(train_scores))
        mean_val.append(np.mean(val_scores))
        mean_test_acc.append(np.mean(test_accs))
        mean_test_f1.append(np.mean(test_f1s))
        mean_test_roc_auc.append(np.mean(test_roc_aucs))

        # Track best test ROC-AUC across all splits/trials
        for idx, t in enumerate(trials):
            if best_model is None or t["test_metrics"]["roc_auc"] > best_model["record"]["test_metrics"]["roc_auc"]:
                best_model = {
                    "split_name": split_name,
                    "trial_index": idx,
                    "record": t,
                }

    # Plot ROC-AUC vs split
    x = np.arange(len(split_names))

    plt.figure()
    plt.plot(split_names, mean_train, marker="o", label="Train (CV mean)")
    plt.plot(split_names, mean_val, marker="o", label="Validation (CV mean)")
    plt.plot(split_names, mean_test_roc_auc, marker="o", label="Test (mean of trials)")
    plt.xlabel("Train/Test split")
    plt.ylabel("ROC-AUC")
    plt.title("Bank Dataset – Boosting ROC-AUC vs split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bank_boosting_accuracy.png"), bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 2. ROC curve for best model (if probabilities available)
    # ------------------------------------------------------------------
    if best_model is not None and best_model["record"]["y_proba"] is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_proba = np.array(rec["y_proba"])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Bank Dataset – Boosting ROC curve (best model)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_boosting_roc_best_model.png"), bbox_inches="tight")
        plt.close()
    else:
        roc_auc = None  # for reporting

    # ------------------------------------------------------------------
    # 3. Confusion matrix for best model
    # ------------------------------------------------------------------
    if best_model is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_pred = np.array(rec["y_pred"])

        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Bank Dataset – Boosting Confusion Matrix (best model)")
        plt.colorbar(im)
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # Add counts on the cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_boosting_confusion_best_model.png"), bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 4. Text report: CV vs Test (overfitting) + best model summary
    # ------------------------------------------------------------------
    report_path = os.path.join(save_dir, "bank_boosting_report.txt")
    with open(report_path, "w") as f:
        f.write("Bank Dataset – Boosting Summary Report\n")
        f.write("======================================\n\n")

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
        f.write("\n(Train vs Val/Test gives a sense of overfitting.\n")
        f.write(" If Train >> Val/Test, model is likely overfitting.)\n\n")

        if best_model is not None:
            split_name = best_model["split_name"]
            idx = best_model["trial_index"]
            rec = best_model["record"]

            f.write("\nBest model (by test ROC-AUC):\n")
            f.write(f"  Split: {split_name}\n")
            f.write(f"  Trial index: {idx}\n")
            f.write(f"  Best params: {rec.get('best_params', {})}\n")

            tm = rec["test_metrics"]
            f.write("\n  Test metrics:\n")
            for k, v in tm.items():
                f.write(f"    {k}: {v:.4f}\n")

            if roc_auc is not None:
                f.write(f"  ROC-AUC used in ROC plot: {roc_auc:.4f}\n")

            # Classification report (precision/recall/F1 per class)
            y_test = np.array(rec["y_test"])
            y_pred = np.array(rec["y_pred"])
            f.write("\nClassification report (test set):\n")
            f.write(classification_report(y_test, y_pred))
        else:
            f.write("No best model found (no trials?)\n")

    print(f"[bank_boosting_plots] Saved accuracy, ROC, confusion matrix plots and report to {save_dir}")