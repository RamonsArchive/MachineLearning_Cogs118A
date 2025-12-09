import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eda_parkinsons(data):
    """
    Exploratory Data Analysis for Parkinson's Telemonitoring dataset.
    
    Task: REGRESSION - Predict total_UPDRS from voice features
    
    Generates:
    - Text report with naive baseline metrics
    - Distribution plots
    - Correlation heatmap
    - Feature vs target scatter plots
    """
    output_dir = "plots/parkinsons_plots/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    df = data.copy()
    
    # Primary target (total_UPDRS is more comprehensive than motor_UPDRS)
    target_col = "total_UPDRS"
    y = df[target_col]
    
    # Features (exclude ID and both targets for correlation analysis)
    exclude_cols = ['subject#', 'motor_UPDRS', 'total_UPDRS']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\n[EDA] Generating Parkinson's EDA...")
    print(f"[EDA] Output directory: {output_dir}")
    
    # ==========================================
    # 1. TEXT REPORT
    # ==========================================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PARKINSON'S TELEMONITORING - EXPLORATORY DATA ANALYSIS")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTask: REGRESSION (Predict UPDRS Score from Voice Features)")
    report_lines.append(f"Target Variable: {target_col}")
    report_lines.append(f"\nUPDRS = Unified Parkinson's Disease Rating Scale")
    report_lines.append(f"Higher score = More severe symptoms")
    
    # Dataset Overview
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 50)
    report_lines.append(f"Total voice recordings: {len(df)}")
    report_lines.append(f"Unique patients: {df['subject#'].nunique()}")
    report_lines.append(f"Recordings per patient: ~{len(df) // df['subject#'].nunique()}")
    report_lines.append(f"Total features: {len(feature_cols)}")
    report_lines.append(f"Voice features: 16 (Jitter, Shimmer, NHR, HNR, etc.)")
    report_lines.append(f"Demographics: age, sex, test_time")
    
    # Target Statistics
    report_lines.append("\n" + "-" * 50)
    report_lines.append("TARGET VARIABLE STATISTICS (total_UPDRS)")
    report_lines.append("-" * 50)
    report_lines.append(f"Mean:  {y.mean():.4f}")
    report_lines.append(f"Std:   {y.std():.4f}")
    report_lines.append(f"Min:   {y.min():.4f}")
    report_lines.append(f"25%:   {y.quantile(0.25):.4f}")
    report_lines.append(f"50%:   {y.median():.4f}")
    report_lines.append(f"75%:   {y.quantile(0.75):.4f}")
    report_lines.append(f"Max:   {y.max():.4f}")
    
    # motor_UPDRS stats too
    motor = df['motor_UPDRS']
    report_lines.append("\n(Alternative target: motor_UPDRS)")
    report_lines.append(f"Mean: {motor.mean():.4f}, Std: {motor.std():.4f}, Range: [{motor.min():.2f}, {motor.max():.2f}]")
    
    # ==========================================
    # NAIVE BASELINE FOR REGRESSION
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("NAIVE BASELINE PERFORMANCE")
    report_lines.append("-" * 50)
    
    y_pred_mean = np.full_like(y, y.mean())
    mse_mean = mean_squared_error(y, y_pred_mean)
    rmse_mean = np.sqrt(mse_mean)
    mae_mean = mean_absolute_error(y, y_pred_mean)
    
    report_lines.append("\nMean Predictor (always predict mean UPDRS):")
    report_lines.append(f"  MSE:  {mse_mean:.4f}")
    report_lines.append(f"  RMSE: {rmse_mean:.4f}")
    report_lines.append(f"  MAE:  {mae_mean:.4f}")
    report_lines.append(f"  R2:   0.0000 (by definition)")
    
    report_lines.append(f"\nA good model should achieve:")
    report_lines.append(f"  RMSE < {rmse_mean:.2f} (beat mean predictor)")
    report_lines.append(f"  R2 > 0.3 for moderate performance")
    report_lines.append(f"  R2 > 0.5 for good performance")
    
    # ==========================================
    # FEATURE CORRELATIONS
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("TOP CORRELATED FEATURES WITH total_UPDRS")
    report_lines.append("-" * 50)
    
    # Compute correlations
    corr_df = df[feature_cols + [target_col]].corr()
    correlations = corr_df[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    
    report_lines.append("\nTop 10 Features (by absolute correlation):")
    for feat, corr in correlations.head(10).items():
        direction = "+" if corr > 0 else "-"
        report_lines.append(f"  {feat}: {direction}{abs(corr):.4f}")
    
    # ==========================================
    # FEATURE GROUPS EXPLANATION
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("VOICE FEATURE GROUPS")
    report_lines.append("-" * 50)
    report_lines.append("""
JITTER (pitch instability):
  - Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP
  - Higher = more pitch variation = worse motor control

SHIMMER (loudness instability):
  - Shimmer, Shimmer(dB), Shimmer:APQ3/5/11, Shimmer:DDA
  - Higher = more amplitude variation = worse motor control

NOISE RATIOS:
  - NHR: Noise-to-Harmonics Ratio (higher = worse voice quality)
  - HNR: Harmonics-to-Noise Ratio (higher = better voice quality)

COMPLEXITY MEASURES:
  - RPDE: Recurrence Period Density Entropy
  - DFA: Detrended Fluctuation Analysis (fractal scaling)
  - PPE: Pitch Period Entropy
    """)
    
    # ==========================================
    # DEMOGRAPHICS
    # ==========================================
    report_lines.append("\n" + "-" * 50)
    report_lines.append("DEMOGRAPHICS")
    report_lines.append("-" * 50)
    report_lines.append(f"\nAge: mean={df['age'].mean():.1f}, range=[{df['age'].min()}, {df['age'].max()}]")
    report_lines.append(f"Sex: {(df['sex']==0).sum()} male (0), {(df['sex']==1).sum()} female (1)")
    report_lines.append(f"Test time: mean={df['test_time'].mean():.1f} days, max={df['test_time'].max():.1f} days")
    
    # Save report
    report_path = os.path.join(output_dir, "eda_parkinsons_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[EDA] Saved report to {report_path}")
    
    # ==========================================
    # 2. PLOTS
    # ==========================================
    
    # --- Target Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
    axes[0].set_xlabel('total_UPDRS Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Distribution: total_UPDRS')
    axes[0].legend()
    
    axes[1].hist(motor, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(motor.mean(), color='red', linestyle='--', label=f'Mean: {motor.mean():.2f}')
    axes[1].set_xlabel('motor_UPDRS Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Alternative Target: motor_UPDRS')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved target_distribution.png")
    
    # --- Correlation Heatmap (top features) ---
    top_features = correlations.abs().head(12).index.tolist()
    corr_subset = df[top_features + [target_col]].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap: Top 12 Features vs total_UPDRS')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved correlation_heatmap.png")
    
    # --- Top Feature Scatter Plots ---
    top_5 = correlations.abs().head(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, feat in enumerate(top_5):
        axes[i].scatter(df[feat], y, alpha=0.3, s=5)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel('total_UPDRS')
        corr_val = correlations[feat]
        axes[i].set_title(f'r = {corr_val:.3f}')
    
    plt.suptitle('Top 5 Correlated Features vs total_UPDRS')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_features_scatter.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved top_features_scatter.png")
    
    # --- UPDRS Progression Over Time ---
    plt.figure(figsize=(10, 5))
    sample_patients = df['subject#'].unique()[:5]  # First 5 patients
    for patient in sample_patients:
        patient_data = df[df['subject#'] == patient].sort_values('test_time')
        plt.plot(patient_data['test_time'], patient_data['total_UPDRS'], 
                 marker='o', markersize=2, alpha=0.7, label=f'Patient {patient}')
    
    plt.xlabel('Days Since Recruitment')
    plt.ylabel('total_UPDRS Score')
    plt.title('UPDRS Progression Over Time (Sample of 5 Patients)')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "updrs_progression.png"), dpi=150)
    plt.close()
    print(f"[EDA] Saved updrs_progression.png")
    
    print(f"\n[EDA] Complete! All outputs saved to {output_dir}/")
    
    return df
