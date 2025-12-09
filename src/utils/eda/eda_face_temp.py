import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eda_face_temp(data):
    """
    Exploratory Data Analysis for Face Temperature dataset (Regression).
    
    Generates:
    - Text report with naive baseline metrics (MSE, RMSE, MAE, R²)
    - Distribution plots
    - Correlation heatmap
    - Feature vs target scatter plots
    """
    output_dir = "plots/face_temp_plots/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # Work on a copy to avoid modifying original
    df = data.copy()
    
    target_col = "OralTemp"
    y = df[target_col]
    
    # Identify feature types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n[EDA] Generating Face Temperature EDA...")
    print(f"[EDA] Output directory: {output_dir}")
    
    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FACE TEMPERATURE DATASET - EXPLORATORY DATA ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTask: REGRESSION (Predict Oral Temperature)")
    report_lines.append(f"Target Variable: {target_col}")
    
    # Dataset Overview
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 50)
    report_lines.append(f"Total samples: {len(df)}")
    report_lines.append(f"Total features: {len(df.columns) - 1}")
    report_lines.append(f"Numeric features: {len(numeric_cols)}")
    report_lines.append(f"Categorical features: {len(cat_cols)}")
    
    # Target Variable Statistics
    report_lines.append("\n" + "-" * 50)
    report_lines.append("TARGET VARIABLE STATISTICS (OralTemp)")
    report_lines.append("-" * 50)
    report_lines.append(f"Mean: {y.mean():.4f}°C")
    report_lines.append(f"Std:  {y.std():.4f}°C")
    report_lines.append(f"Min:  {y.min():.4f}°C")
    report_lines.append(f"25%:  {y.quantile(0.25):.4f}°C")
    report_lines.append(f"50%:  {y.median():.4f}°C")
    report_lines.append(f"75%:  {y.quantile(0.75):.4f}°C")
    report_lines.append(f"Max:  {y.max():.4f}°C")
    
    # ==========================================
    # NAIVE BASELINE FOR REGRESSION
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("NAIVE BASELINE PERFORMANCE")
    report_lines.append("-" * 50)
    report_lines.append("\nFor regression, naive baselines predict a constant value:")
    report_lines.append("  - Mean Predictor: Always predict the mean of target")
    report_lines.append("  - Median Predictor: Always predict the median of target")
    
    # Mean predictor baseline
    y_pred_mean = np.full_like(y, y.mean())
    mse_mean = mean_squared_error(y, y_pred_mean)
    rmse_mean = np.sqrt(mse_mean)
    mae_mean = mean_absolute_error(y, y_pred_mean)
    r2_mean = r2_score(y, y_pred_mean)  # Always 0 for mean predictor
    
    report_lines.append("\n1. MEAN PREDICTOR (predict mean = {:.4f}°C):".format(y.mean()))
    report_lines.append(f"   MSE:  {mse_mean:.6f}")
    report_lines.append(f"   RMSE: {rmse_mean:.6f}°C")
    report_lines.append(f"   MAE:  {mae_mean:.6f}°C")
    report_lines.append(f"   R²:   {r2_mean:.6f} (always 0 for mean predictor)")
    
    # Median predictor baseline
    y_pred_median = np.full_like(y, y.median())
    mse_median = mean_squared_error(y, y_pred_median)
    rmse_median = np.sqrt(mse_median)
    mae_median = mean_absolute_error(y, y_pred_median)
    r2_median = r2_score(y, y_pred_median)
    
    report_lines.append("\n2. MEDIAN PREDICTOR (predict median = {:.4f}°C):".format(y.median()))
    report_lines.append(f"   MSE:  {mse_median:.6f}")
    report_lines.append(f"   RMSE: {rmse_median:.6f}°C")
    report_lines.append(f"   MAE:  {mae_median:.6f}°C")
    report_lines.append(f"   R²:   {r2_median:.6f}")
    
    report_lines.append("\n" + "-" * 50)
    report_lines.append("BASELINE INTERPRETATION")
    report_lines.append("-" * 50)
    report_lines.append(f"\nA good model should achieve:")
    report_lines.append(f"  - RMSE < {rmse_mean:.4f}°C (beat mean predictor)")
    report_lines.append(f"  - MAE  < {mae_mean:.4f}°C (beat mean predictor)")
    report_lines.append(f"  - R²   > 0 (explain some variance)")
    report_lines.append(f"\nTypical 'good' regression performance:")
    report_lines.append(f"  - R² > 0.5 (explains 50%+ of variance)")
    report_lines.append(f"  - R² > 0.7 (good model)")
    report_lines.append(f"  - R² > 0.9 (excellent model)")
    
    # ==========================================
    # TOP CORRELATED FEATURES
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("TOP CORRELATED FEATURES WITH TARGET")
    report_lines.append("-" * 50)
    
    # Compute correlations with target
    correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    top_positive = correlations.nlargest(10)
    top_negative = correlations.nsmallest(5)
    
    report_lines.append("\nTop 10 Positive Correlations:")
    for feat, corr in top_positive.items():
        report_lines.append(f"  {feat}: {corr:.4f}")
    
    report_lines.append("\nTop 5 Negative Correlations:")
    for feat, corr in top_negative.items():
        report_lines.append(f"  {feat}: {corr:.4f}")
    
    # ==========================================
    # CATEGORICAL FEATURE ANALYSIS
    # ==========================================
    if cat_cols:
        report_lines.append("\n" + "-" * 50)
        report_lines.append("CATEGORICAL FEATURE ANALYSIS")
        report_lines.append("-" * 50)
        
        for col in cat_cols:
            report_lines.append(f"\n{col} - Mean OralTemp by category:")
            cat_means = df.groupby(col)[target_col].agg(['mean', 'std', 'count'])
            for idx, row in cat_means.iterrows():
                report_lines.append(f"  {idx}: {row['mean']:.4f}°C (±{row['std']:.4f}, n={int(row['count'])})")
    
    # Save text report (with UTF-8 encoding for special characters)
    report_path = os.path.join(output_dir, "eda_face_temp_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[EDA] Saved report to {report_path}")
    
    # ==========================================
    # 2. PLOTS
    # ==========================================
    
    # --- Target Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(y, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}°C')
    axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Median: {y.median():.2f}°C')
    axes[0].set_xlabel('Oral Temperature (°C)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Distribution: Oral Temperature')
    axes[0].legend()
    
    axes[1].boxplot(y, vert=True)
    axes[1].set_ylabel('Oral Temperature (°C)')
    axes[1].set_title('Target Boxplot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved target_distribution.png")
    
    # --- Correlation Heatmap (top features only) ---
    top_features = correlations.abs().nlargest(15).index.tolist()
    if len(top_features) > 0:
        corr_subset = df[top_features + [target_col]].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Heatmap: Top 15 Features vs Target')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
        plt.close()
        print(f"[EDA] Saved correlation_heatmap.png")
    
    # --- Top Feature Scatter Plots ---
    top_5_features = correlations.abs().nlargest(5).index.tolist()
    if len(top_5_features) > 0:
        fig, axes = plt.subplots(1, min(5, len(top_5_features)), figsize=(15, 3))
        if len(top_5_features) == 1:
            axes = [axes]
        
        for i, feat in enumerate(top_5_features[:5]):
            axes[i].scatter(df[feat], y, alpha=0.5, s=10)
            axes[i].set_xlabel(feat[:20] + '...' if len(feat) > 20 else feat)
            axes[i].set_ylabel('OralTemp (°C)')
            
            # Add trend line
            z = np.polyfit(df[feat].dropna(), y[df[feat].notna()], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
            axes[i].plot(x_line, p(x_line), "r--", alpha=0.8)
            
            corr_val = correlations[feat]
            axes[i].set_title(f'r = {corr_val:.3f}')
        
        plt.suptitle('Top 5 Correlated Features vs Oral Temperature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_features_scatter.png"), dpi=150)
        plt.close()
        print(f"[EDA] Saved top_features_scatter.png")
    
    # --- Categorical Boxplots ---
    if cat_cols:
        for col in cat_cols[:3]:  # Limit to first 3 categorical
            plt.figure(figsize=(10, 5))
            df.boxplot(column=target_col, by=col)
            plt.title(f'Oral Temperature by {col}')
            plt.suptitle('')  # Remove automatic title
            plt.xlabel(col)
            plt.ylabel('Oral Temperature (°C)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"), dpi=150)
            plt.close()
            print(f"[EDA] Saved boxplot_{col}.png")
    
    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")
    
    return df
