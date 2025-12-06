# src/graphs/bank_svm_plots.py
"""
Plotting utilities for SVM model results on Bank Marketing dataset.
Generates:
- Accuracy comparison across splits
- ROC curve for best model
- Confusion matrix for best model
- Kernel comparison chart
- Summary report comparing all splits
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


def plot_bank_svm_summary(results_svm, save_dir):
    """
    results_svm: results["svm"] sub-dict from bank.py
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
      - "svm_info": dict with kernel, C, gamma, support vectors info
    """

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Accuracy vs split (mean over 3 trials)
    # ------------------------------------------------------------------
    split_names = []
    mean_train = []
    mean_val = []
    mean_test = []

    # Also keep overall-best model for ROC/confusion matrix
    best_model = None

    for split_name, trials in results_svm.items():
        split_names.append(split_name)

        train_scores = [t["cv_train_score"] for t in trials]
        val_scores = [t["cv_val_score"] for t in trials]
        test_accs = [t["test_accuracy"] for t in trials]

        mean_train.append(np.mean(train_scores))
        mean_val.append(np.mean(val_scores))
        mean_test.append(np.mean(test_accs))

        # Track best test accuracy across all splits/trials
        for idx, t in enumerate(trials):
            if best_model is None or t["test_accuracy"] > best_model["record"]["test_accuracy"]:
                best_model = {
                    "split_name": split_name,
                    "trial_index": idx,
                    "record": t,
                }

    # Plot accuracy vs split
    plt.figure(figsize=(8, 5))
    plt.plot(split_names, mean_train, marker="o", linewidth=2, markersize=8, label="Train (CV mean)")
    plt.plot(split_names, mean_val, marker="s", linewidth=2, markersize=8, label="Validation (CV mean)")
    plt.plot(split_names, mean_test, marker="^", linewidth=2, markersize=8, label="Test (mean of trials)")
    plt.xlabel("Train/Test Split", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Bank Dataset – SVM: Performance vs Train/Test Split", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bank_svm_accuracy.png"), dpi=150, bbox_inches="tight")
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
        plt.plot(fpr, tpr, color="#059669", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
        plt.fill_between(fpr, tpr, alpha=0.2, color="#059669")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title(f"Bank Dataset – SVM ROC Curve (Best Model)\nSplit: {best_model['split_name']}, Trial: {best_model['trial_index']}", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_svm_roc_best_model.png"), dpi=150, bbox_inches="tight")
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
        im = plt.imshow(cm, interpolation="nearest", cmap="Greens")
        plt.title(f"Bank Dataset – SVM Confusion Matrix\n(Best Model: {best_model['split_name']})", fontsize=12)
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
        plt.savefig(os.path.join(save_dir, "bank_svm_confusion_best_model.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 4. Kernel comparison (if multiple kernels tested)
    # ------------------------------------------------------------------
    kernel_performance = {}
    for split_name, trials in results_svm.items():
        for t in trials:
            kernel = t.get("svm_info", {}).get("kernel", "unknown")
            if kernel not in kernel_performance:
                kernel_performance[kernel] = []
            kernel_performance[kernel].append(t["test_accuracy"])
    
    if len(kernel_performance) > 1:
        kernel_names = list(kernel_performance.keys())
        kernel_means = [np.mean(kernel_performance[k]) for k in kernel_names]
        kernel_stds = [np.std(kernel_performance[k]) for k in kernel_names]
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(kernel_names, kernel_means, yerr=kernel_stds, 
                      color="#059669", alpha=0.8, capsize=5)
        plt.xlabel("Kernel", fontsize=11)
        plt.ylabel("Mean Test Accuracy", fontsize=11)
        plt.title("Bank Dataset – SVM Kernel Comparison", fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, kernel_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bank_svm_kernel_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 5. Text report
    # ------------------------------------------------------------------
    report_path = os.path.join(save_dir, "bank_svm_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Bank Dataset – Support Vector Machine (SVM) Summary Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write("Model: Support Vector Classifier (SVC)\n")
        f.write("Evaluation: 10-fold Cross-Validation + Held-out Test Set\n")
        f.write("Scoring Metric: ROC-AUC\n\n")

        f.write("SVM HYPERPARAMETERS GUIDE:\n")
        f.write("  • kernel: 'rbf' (flexible) vs 'linear' (faster)\n")
        f.write("  • C: Regularization (higher = tighter fit)\n")
        f.write("  • gamma: Kernel coefficient (higher = complex boundaries)\n\n")

        f.write("PERFORMANCE BY SPLIT (averaged over 3 trials)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Split':<12} {'Train (CV)':<15} {'Val (CV)':<15} {'Test':<15}\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            f.write(
                f"{split_name:<12} {mean_train[i]:<15.4f} {mean_val[i]:<15.4f} {mean_test[i]:<15.4f}\n"
            )
        f.write("\n")

        # Overfitting analysis
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for i, split_name in enumerate(split_names):
            overfit_gap = mean_train[i] - mean_val[i]
            generalization_gap = mean_val[i] - mean_test[i]
            f.write(f"Split {split_name}:\n")
            f.write(f"  Train-Val gap: {overfit_gap:+.4f} (>0.05 suggests overfitting)\n")
            f.write(f"  Val-Test gap:  {generalization_gap:+.4f}\n")
        f.write("\n")

        if best_model is not None:
            split_name = best_model["split_name"]
            idx = best_model["trial_index"]
            rec = best_model["record"]

            f.write("BEST MODEL (by test accuracy)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Split: {split_name}\n")
            f.write(f"Trial: {idx}\n")
            f.write(f"\nBest Hyperparameters:\n")
            for param, val in rec.get('best_params', {}).items():
                clean_param = param.replace("model__", "")
                f.write(f"  {clean_param}: {val}\n")

            # SVM-specific info
            svm_info = rec.get("svm_info", {})
            if svm_info:
                f.write(f"\nSVM Details:\n")
                f.write(f"  Kernel: {svm_info.get('kernel', 'N/A')}\n")
                f.write(f"  C (regularization): {svm_info.get('C', 'N/A')}\n")
                f.write(f"  gamma: {svm_info.get('gamma', 'N/A')}\n")
                if svm_info.get('total_support_vectors'):
                    f.write(f"  Total Support Vectors: {svm_info['total_support_vectors']}\n")
                    f.write(f"  Support Vectors per class: {svm_info.get('n_support', 'N/A')}\n")

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

            # Classification report
            y_test = np.array(rec["y_test"])
            y_pred = np.array(rec["y_pred"])
            f.write("\nDetailed Classification Report (Test Set):\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))
        else:
            f.write("No best model found (no trials?)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[bank_svm_plots] Saved plots and report to {save_dir}")

