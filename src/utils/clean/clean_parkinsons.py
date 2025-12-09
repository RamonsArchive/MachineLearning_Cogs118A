import pandas as pd
import numpy as np


def clean_parkinsons(data):
    """
    Clean Parkinson's telemonitoring dataset.
    
    Dataset has NO missing values, so cleaning is minimal:
    1. Quick null check (confirm no missing values)
    2. Drop subject# (ID column, not a feature)
    3. Choose target variable (motor_UPDRS or total_UPDRS)
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for modeling
    """
    print("\n" + "="*60)
    print("CLEANING PARKINSON'S TELEMONITORING DATA")
    print("="*60)
    
    df = data.copy()
    
    # ==========================================
    # 1. Quick Null Check (should be 0)
    # ==========================================
    print("\n--- Null Value Check ---")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        print("✓ No missing values (as expected)")
    else:
        print(f"⚠ Found {total_nulls} null values:")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print(f"Dropped rows with nulls. New shape: {df.shape}")
    
    # ==========================================
    # 2. Data Overview
    # ==========================================
    print("\n--- Dataset Overview ---")
    print(f"Total recordings: {len(df)}")
    print(f"Unique patients: {df['subject#'].nunique()}")
    print(f"Recordings per patient: ~{len(df) // df['subject#'].nunique()}")
    print(f"Features: {len(df.columns) - 3} voice measures + age, sex, test_time")
    
    # ==========================================
    # 3. Target Variables
    # ==========================================
    print("\n--- Target Variables (UPDRS Scores) ---")
    print(f"motor_UPDRS: min={df['motor_UPDRS'].min():.2f}, max={df['motor_UPDRS'].max():.2f}, mean={df['motor_UPDRS'].mean():.2f}")
    print(f"total_UPDRS: min={df['total_UPDRS'].min():.2f}, max={df['total_UPDRS'].max():.2f}, mean={df['total_UPDRS'].mean():.2f}")
    print("\nWe'll use total_UPDRS as the primary target (more comprehensive)")
    
    # ==========================================
    # 4. Drop ID Column
    # ==========================================
    # subject# is an ID, not a predictor
    # But we keep it for now in case we want to do per-patient analysis
    # We'll drop it in the experiment when defining predictors
    
    print("\n--- Final Dataset ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df
