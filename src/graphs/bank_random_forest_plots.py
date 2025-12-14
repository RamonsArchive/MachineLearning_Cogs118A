# src/graphs/bank_random_forest_plots.py
"""
Plotting utilities for Random Forest model results on Bank Marketing dataset.
Generates:
- Accuracy comparison across splits
- ROC curve for best model
- Confusion matrix for best model  
- Feature importance bar chart
- Summary report comparing all splits
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def plot_bank_random_forest_summary(results_rf, save_dir):
    """
    results_rf: results["random_forest"] sub-dict from bank.py
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
      - "feature_importances": list of floats or None
      - "feature_names": list of strings or None
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

    for split_name, trials in results_rf.items():
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
    plt.figure(figsize=(8, 5))
    plt.plot(split_names, mean_train, marker="o", linewidth=2, markersize=8, label="Train (CV mean)")
    plt.plot(split_names, mean_val, marker="s", linewidth=2, markersize=8, label="Validation (CV mean)")
    plt.plot(split_names, mean_test_roc_auc, marker="^", linewidth=2, markersize=8, label="Test (mean of trials)")
    plt.xlabel("Train/Test Split", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Bank Dataset – Random Forest: Performance vs Train/Test Split", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bank_rf_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 2. ROC curve for best model (if probabilities available)
    # ------------------------------------------------------------------
    roc_auc = None
    if best_model is not None and best_model["record"]["y_proba"] is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_proba = np.array(rec["y_proba"])

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
        plt.fill_between(fpr, tpr, alpha=0.2, color="#2563eb")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(f"Bank Dataset – Random Forest ROC Curve (Best Model)\nSplit: {best_model['split_name']}, Trial: {best_model['trial_index']}", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_rf_roc_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 3. Confusion matrix for best model
    # ------------------------------------------------------------------
    if best_model is not None:
        rec = best_model["record"]
        y_test = np.array(rec["y_test"])
        y_pred = np.array(rec["y_pred"])

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Bank Dataset – Random Forest Confusion Matrix\n(Best Model: {best_model['split_name']})", fontsize=12)
        plt.colorbar(im)
        tick_marks = np.arange(cm.shape[0])
        class_labels = ["No (0)", "Yes (1)"]
        plt.xticks(tick_marks, class_labels, fontsize=10)
        plt.yticks(tick_marks, class_labels, fontsize=10)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)

        # Add counts on the cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_rf_confusion_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 4. Feature importance bar chart (for best model)
    # ------------------------------------------------------------------
    if best_model is not None:
        rec = best_model["record"]
        feat_importances = rec.get("feature_importances")
        feat_names = rec.get("feature_names")
        
        if feat_importances is not None and feat_names is not None:
            # Sort by importance and take top 15
            sorted_idx = np.argsort(feat_importances)[::-1][:15]
            top_importances = [feat_importances[i] for i in sorted_idx]
            top_names = [feat_names[i] for i in sorted_idx]
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(top_names)), top_importances[::-1], color="#2563eb", alpha=0.8)
            plt.yticks(range(len(top_names)), top_names[::-1], fontsize=9)
            plt.xlabel("Feature Importance", fontsize=11)
            plt.title(f"Bank Dataset – Random Forest Top 15 Feature Importances\n(Best Model: {best_model['split_name']})", fontsize=12)
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "bank_rf_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------------
    # 5. Text report: CV vs Test (overfitting) + best model summary
    # ------------------------------------------------------------------
    report_path = os.path.join(save_dir, "bank_random_forest_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Bank Dataset – Random Forest Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write("Model: Random Forest Classifier\n")
        f.write("Evaluation: 10-fold Cross-Validation + Held-out Test Set\n")
        f.write("Scoring Metric: ROC-AUC\n\n")

        f.write("PERFORMANCE BY SPLIT (averaged over 3 trials)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Split':<12} {'CV Train':<12} {'CV Val':<12} {'Test Acc':<12} {'Test F1':<12} {'Test ROC-AUC':<12}\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            f.write(
                f"{split_name:<12} "
                f"{mean_train[i]:<12.4f} "
                f"{mean_val[i]:<12.4f} "
                f"{mean_test_acc[i]:<12.4f} "
                f"{mean_test_f1[i]:<12.4f} "
                f"{mean_test_roc_auc[i]:<12.4f}\n"
            )
        f.write("\n")

        # Overfitting analysis
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            overfit_gap = mean_train[i] - mean_val[i]
            generalization_gap = mean_val[i] - mean_test_roc_auc[i]
            f.write(f"Split {split_name}:\n")
            f.write(f"  Train-Val gap: {overfit_gap:+.4f} (>0.05 suggests overfitting)\n")
            f.write(f"  Val-Test gap:  {generalization_gap:+.4f} (large gap = CV overestimates performance)\n")
        f.write("\n")

        if best_model is not None:
            split_name = best_model["split_name"]
            idx = best_model["trial_index"]
            rec = best_model["record"]

            f.write("BEST MODEL (by test ROC-AUC)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Split: {split_name}\n")
            f.write(f"Trial: {idx}\n")
            f.write(f"\nBest Hyperparameters:\n")
            for param, val in rec.get('best_params', {}).items():
                # Clean up param name for readability
                clean_param = param.replace("model__", "")
                f.write(f"  {clean_param}: {val}\n")

            tm = rec["test_metrics"]
            f.write("\nTest Set Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Accuracy:  {tm.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision: {tm.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {tm.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:  {tm.get('f1', 0):.4f}\n")
            f.write(f"  ROC-AUC:   {tm.get('roc_auc', 0):.4f}\n")

            if roc_auc is not None:
                f.write(f"\n(ROC-AUC from ROC curve plot: {roc_auc:.4f})\n")

            # Classification report (precision/recall/F1 per class)
            y_test = np.array(rec["y_test"])
            y_pred = np.array(rec["y_pred"])
            f.write("\nDetailed Classification Report (Test Set):\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

            # Feature importances (top 10)
            feat_importances = rec.get("feature_importances")
            feat_names = rec.get("feature_names")
            if feat_importances is not None and feat_names is not None:
                f.write("\nTop 10 Most Important Features:\n")
                f.write("-" * 40 + "\n")
                sorted_idx = np.argsort(feat_importances)[::-1][:10]
                for rank, i in enumerate(sorted_idx, 1):
                    f.write(f"  {rank:2}. {feat_names[i]:<30} {feat_importances[i]:.4f}\n")
        else:
            f.write("No best model found (no trials?)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[bank_random_forest_plots] Saved plots and report to {save_dir}")

