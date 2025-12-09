"""
Plotting functions for Parkinson's Telemonitoring Regression experiments.
Generates summary reports, scatter plots, and comparison charts.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _compute_split_averages(split_results):
    """Compute average metrics across trials for a split."""
    if not split_results:
        return {}
    
    metrics = {
        "cv_train_score": [],
        "cv_val_score": [],
        "test_mse": [],
        "test_rmse": [],
        "test_mae": [],
        "test_r2": [],
    }
    
    for trial in split_results:
        metrics["cv_train_score"].append(trial["cv_train_score"])
        metrics["cv_val_score"].append(trial["cv_val_score"])
        metrics["test_mse"].append(trial["test_metrics"]["mse"])
        metrics["test_rmse"].append(trial["test_metrics"]["rmse"])
        metrics["test_mae"].append(trial["test_metrics"]["mae"])
        metrics["test_r2"].append(trial["test_metrics"]["r2"])
    
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def _generate_regression_report(results, model_name, output_dir):
    """Generate text report for a regression model."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"PARKINSON'S TELEMONITORING - {model_name.upper()} RESULTS")
    report_lines.append("=" * 70)
    
    # Performance by Split
    report_lines.append("\n" + "-" * 50)
    report_lines.append("PERFORMANCE BY SPLIT (averaged over 3 trials)")
    report_lines.append("-" * 50)
    report_lines.append(f"\n{'Split':<10} {'CV Train':<12} {'CV Val':<12} {'Test R2':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    report_lines.append("-" * 70)
    
    best_r2 = -float('inf')
    best_split = None
    best_trial = None
    
    for split_name, trials in results.items():
        avg = _compute_split_averages(trials)
        
        report_lines.append(
            f"{split_name:<10} "
            f"{avg['cv_train_score'][0]:.4f}      "
            f"{avg['cv_val_score'][0]:.4f}      "
            f"{avg['test_r2'][0]:.4f}      "
            f"{avg['test_rmse'][0]:.4f}       "
            f"{avg['test_mae'][0]:.4f}"
        )
        
        # Find best trial
        for trial in trials:
            if trial["test_metrics"]["r2"] > best_r2:
                best_r2 = trial["test_metrics"]["r2"]
                best_split = split_name
                best_trial = trial
    
    # Best Model Details
    if best_trial:
        report_lines.append("\n" + "-" * 50)
        report_lines.append("BEST MODEL (highest R2)")
        report_lines.append("-" * 50)
        report_lines.append(f"Split: {best_split}")
        report_lines.append(f"Trial: {best_trial['trial'] + 1}")
        report_lines.append(f"\nBest Parameters:")
        for param, val in best_trial["best_params"].items():
            report_lines.append(f"  {param}: {val}")
        
        report_lines.append(f"\nTest Metrics:")
        report_lines.append(f"  R2:   {best_trial['test_metrics']['r2']:.4f}")
        report_lines.append(f"  RMSE: {best_trial['test_metrics']['rmse']:.4f} degC")
        report_lines.append(f"  MAE:  {best_trial['test_metrics']['mae']:.4f} degC")
        report_lines.append(f"  MSE:  {best_trial['test_metrics']['mse']:.6f}")
    
    # Interpretation
    report_lines.append("\n" + "-" * 50)
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 50)
    if best_trial:
        r2 = best_trial['test_metrics']['r2']
        if r2 > 0.9:
            quality = "EXCELLENT"
        elif r2 > 0.7:
            quality = "GOOD"
        elif r2 > 0.5:
            quality = "MODERATE"
        else:
            quality = "POOR"
        
        report_lines.append(f"Model Quality: {quality} (R2 = {r2:.4f})")
        report_lines.append(f"The model explains {r2*100:.1f}% of variance in UPDRS scores.")
        report_lines.append(f"Average prediction error: +/- {best_trial['test_metrics']['mae']:.3f} degC")
    
    # Save report
    report_path = os.path.join(output_dir, f"parkinsons_{model_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    return best_trial


def _plot_predictions_scatter(best_trial, model_name, output_dir):
    """Plot predicted vs actual scatter plot."""
    y_test = np.array(best_trial["y_test"])
    y_pred = np.array(best_trial["y_pred"])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, c='steelblue')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual total_UPDRS (°C)', fontsize=12)
    ax.set_ylabel('Predicted total_UPDRS (°C)', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs Actual\nR² = {best_trial["test_metrics"]["r2"]:.4f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"parkinsons_{model_name}_scatter.png"), dpi=150)
    plt.close()


def _plot_residuals(best_trial, model_name, output_dir):
    """Plot residuals histogram and distribution."""
    y_test = np.array(best_trial["y_test"])
    y_pred = np.array(best_trial["y_pred"])
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[0].axvline(residuals.mean(), color='orange', linestyle='--', lw=2, 
                    label=f'Mean: {residuals.mean():.4f}')
    axes[0].set_xlabel('Residual (Actual - Predicted) °C', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Residuals Distribution', fontsize=14)
    axes[0].legend()
    
    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30, c='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted total_UPDRS (°C)', fontsize=12)
    axes[1].set_ylabel('Residual (°C)', fontsize=12)
    axes[1].set_title('Residuals vs Predicted Values', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"parkinsons_{model_name}_residuals.png"), dpi=150)
    plt.close()


def _plot_r2_by_split(results, model_name, output_dir):
    """Plot R2 scores across splits."""
    splits = list(results.keys())
    r2_means = []
    r2_stds = []
    
    for split_name in splits:
        avg = _compute_split_averages(results[split_name])
        r2_means.append(avg["test_r2"][0])
        r2_stds.append(avg["test_r2"][1])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(splits))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Train/Test Split', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{model_name}: R² by Split (± std over 3 trials)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"parkinsons_{model_name}_r2_by_split.png"), dpi=150)
    plt.close()


# ==========================================
# Public API
# ==========================================

def plot_parkinsons_boosting_summary(results, output_dir):
    """Generate all plots and report for Boosting (XGBoost) results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "boosting"
    
    print(f"[plot] Generating {model_name} (XGBoost) plots to {output_dir}")
    
    best_trial = _generate_regression_report(results, model_name, output_dir)
    
    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        _plot_residuals(best_trial, model_name, output_dir)
        
        # Feature importance plot
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)
    
    _plot_r2_by_split(results, model_name, output_dir)
    
    print(f"[plot] Saved {model_name} plots and report")


def plot_parkinsons_random_forest_summary(results, output_dir):
    """Generate all plots and report for Random Forest results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "random_forest"
    
    print(f"[plot] Generating {model_name} plots to {output_dir}")
    
    best_trial = _generate_regression_report(results, model_name, output_dir)
    
    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        _plot_residuals(best_trial, model_name, output_dir)
        
        # Feature importance plot (RF specific)
        if best_trial.get("feature_importances") and best_trial.get("feature_names"):
            _plot_feature_importances(best_trial, model_name, output_dir)
    
    _plot_r2_by_split(results, model_name, output_dir)
    
    print(f"[plot] Saved {model_name} plots and report")


def _plot_feature_importances(best_trial, model_name, output_dir):
    """Plot top feature importances for Random Forest."""
    importances = np.array(best_trial["feature_importances"])
    feature_names = best_trial["feature_names"]
    
    # Get top 15 features
    indices = np.argsort(importances)[-15:][::-1]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, color='steelblue', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"parkinsons_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


def _plot_ridge_coefficients(best_trial, model_name, output_dir):
    """Plot top Ridge regression coefficients (shows linear relationships)."""
    coefficients = np.array(best_trial["coefficients"])
    feature_names = best_trial["feature_names"]
    
    # Get top 15 by absolute value
    abs_coefs = np.abs(coefficients)
    indices = np.argsort(abs_coefs)[-15:][::-1]
    top_coefs = coefficients[indices]
    top_names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    colors = ['forestgreen' if c > 0 else 'coral' for c in top_coefs]
    ax.barh(y_pos, top_coefs, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_title('Top 15 Ridge Coefficients\n(Green=positive, Red=negative)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"parkinsons_{model_name}_coefficients.png"), dpi=150)
    plt.close()


def plot_parkinsons_neural_network_summary(results, output_dir):
    """Generate all plots and report for Neural Network results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "neural_network"
    
    print(f"[plot] Generating {model_name} plots to {output_dir}")
    
    best_trial = _generate_regression_report(results, model_name, output_dir)
    
    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        _plot_residuals(best_trial, model_name, output_dir)
    
    _plot_r2_by_split(results, model_name, output_dir)
    
    print(f"[plot] Saved {model_name} plots and report")


def plot_parkinsons_elastic_net_summary(results, output_dir):
    """Generate all plots and report for ElasticNet (linear baseline) results."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = "elastic_net"
    
    print(f"[plot] Generating {model_name} (linear baseline) plots to {output_dir}")
    
    best_trial = _generate_regression_report(results, model_name, output_dir)
    
    if best_trial:
        _plot_predictions_scatter(best_trial, model_name, output_dir)
        _plot_residuals(best_trial, model_name, output_dir)
        
        # Feature coefficients plot (shows which features matter for linear model)
        if best_trial.get("coefficients") is not None and best_trial.get("feature_names"):
            _plot_ridge_coefficients(best_trial, model_name, output_dir)
    
    _plot_r2_by_split(results, model_name, output_dir)
    
    print(f"[plot] Saved {model_name} plots and report")


def plot_parkinsons_model_comparison(all_results, output_dir):
    """Generate comparison plots across all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[plot] Generating model comparison plots to {output_dir}")
    
    model_names = list(all_results.keys())
    splits = list(all_results[model_names[0]].keys())
    
    # ==========================================
    # R2 Comparison Bar Chart
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(splits))
    width = 0.2  # Narrower for 4 models
    
    # Colors for 4 models: Ridge (baseline), Boosting, RF, NN
    colors = ['gray', 'steelblue', 'forestgreen', 'coral']
    
    for i, model in enumerate(model_names):
        r2_means = []
        r2_stds = []
        for split in splits:
            avg = _compute_split_averages(all_results[model][split])
            r2_means.append(avg["test_r2"][0])
            r2_stds.append(avg["test_r2"][1])
        
        offset = (i - 1.5) * width  # Center 4 bars
        bars = ax.bar(x + offset, r2_means, width, yerr=r2_stds, capsize=3,
                      label=model.replace('_', ' ').title(), color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Train/Test Split', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Model Comparison: R² by Split (± std over 3 trials)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_r2.png"), dpi=150)
    plt.close()
    
    # ==========================================
    # RMSE Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(model_names):
        rmse_means = []
        rmse_stds = []
        for split in splits:
            avg = _compute_split_averages(all_results[model][split])
            rmse_means.append(avg["test_rmse"][0])
            rmse_stds.append(avg["test_rmse"][1])
        
        offset = (i - 1.5) * width  # Center 4 bars
        ax.bar(x + offset, rmse_means, width, yerr=rmse_stds, capsize=3,
               label=model.replace('_', ' ').title(), color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Train/Test Split', fontsize=12)
    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title('Model Comparison: RMSE by Split (lower is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_rmse.png"), dpi=150)
    plt.close()
    
    # ==========================================
    # Summary Table
    # ==========================================
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PARKINSON'S TELEMONITORING - MODEL COMPARISON SUMMARY")
    report_lines.append("=" * 80)
    
    report_lines.append("\n" + "-" * 60)
    report_lines.append("AVERAGE TEST R2 BY MODEL AND SPLIT")
    report_lines.append("-" * 60)
    
    header = f"{'Model':<20}"
    for split in splits:
        header += f"{split:<15}"
    header += f"{'Overall':<15}"
    report_lines.append(header)
    report_lines.append("-" * 60)
    
    for model in model_names:
        row = f"{model:<20}"
        all_r2 = []
        for split in splits:
            avg = _compute_split_averages(all_results[model][split])
            r2 = avg["test_r2"][0]
            all_r2.append(r2)
            row += f"{r2:.4f}         "
        row += f"{np.mean(all_r2):.4f}"
        report_lines.append(row)
    
    # Best model
    report_lines.append("\n" + "-" * 60)
    report_lines.append("BEST OVERALL MODEL")
    report_lines.append("-" * 60)
    
    best_model = None
    best_r2 = -float('inf')
    
    for model in model_names:
        for split in splits:
            for trial in all_results[model][split]:
                if trial["test_metrics"]["r2"] > best_r2:
                    best_r2 = trial["test_metrics"]["r2"]
                    best_model = model
                    best_split = split
                    best_trial_info = trial
    
    if best_model:
        report_lines.append(f"Model: {best_model}")
        report_lines.append(f"Split: {best_split}")
        report_lines.append(f"R2:    {best_r2:.4f}")
        report_lines.append(f"RMSE:  {best_trial_info['test_metrics']['rmse']:.4f} degC")
        report_lines.append(f"MAE:   {best_trial_info['test_metrics']['mae']:.4f} degC")
    
    # Save comparison report
    report_path = os.path.join(output_dir, "model_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[plot] Saved model comparison plots and report")

