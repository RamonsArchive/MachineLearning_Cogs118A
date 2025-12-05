import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

def eda_bank(data):
    # Create eda_plots directory if it doesn't exist
    output_dir = "plots/eda_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # ----- Save text report with info and describe -----
    report_path = os.path.join(output_dir, "eda_bank_plots.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BANK DATA EDA REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATA INFO:\n")
        f.write("-" * 80 + "\n")
        buffer = StringIO()
        data.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\n")
        
        f.write("DATA DESCRIBE:\n")
        f.write("-" * 80 + "\n")
        f.write(str(data.describe()))
        f.write("\n\n")
        
        # Encode y for correlation
        numeric_cols = data.select_dtypes(include=['int64','float64']).columns
        data["_y"] = (data["y"] == "yes").astype(int)
        corr_with_target = data[numeric_cols.tolist() + ["_y"]].corr()["_y"].sort_values(ascending=False)
        f.write("CORRELATION OF NUMERIC FEATURES WITH TARGET:\n")
        f.write("-" * 80 + "\n")
        f.write(str(corr_with_target))
        f.write("\n")
    
    print(f"Text report saved to {report_path}")

    # ----- class inbalance -----
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=data)
    plt.title("Class Distribution (Subscription)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_1.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 1 saved: eda_bank_plots_1.png")

    # ----- Correlation Heatmap (numeric only) -----
    numeric_cols = data.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(10,8))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_2.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 2 saved: eda_bank_plots_2.png")

    # ----- Boxplot for strongest predictor -----
    plt.figure(figsize=(8,6))
    sns.boxplot(x="y", y="duration", data=data)
    plt.title("Call Duration by Subscription Outcome")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_3.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 3 saved: eda_bank_plots_3.png")

    # ----- Category Subscription Rates -----
    plt.figure(figsize=(10,6))
    (data.groupby("job")["y"]
         .value_counts(normalize=True)
         .unstack()["yes"]
         .sort_values()
         .plot(kind="barh"))
    plt.title("Subscription Rate by Job Type")
    plt.xlabel("Rate of Subscription (yes)")
    plt.savefig(os.path.join(output_dir, "eda_bank_plots_4.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot 4 saved: eda_bank_plots_4.png")
    
    print(f"\nAll plots and report saved to {output_dir}/ directory")

