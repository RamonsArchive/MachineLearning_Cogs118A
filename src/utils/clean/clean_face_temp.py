import pandas as pd
import numpy as np
import re


def clean_face_temp(ici_data, flir_data):
    """
    Clean and prepare face temperature data for regression.
    
    Strategy:
    1. Drop spacer columns (Unnamed:*)
    2. Average temperature measurements across all available rounds (1-4)
    3. Create single OralTemp target based on Gender
    4. Merge ICI and FLIR sensor data by SubjectID
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for modeling
    """
    print("\n" + "="*60)
    print("CLEANING FACE TEMPERATURE DATA")
    print("="*60)
    
    # ==========================================
    # 1. Process ICI Sensor Data
    # ==========================================
    ici_clean = _process_sensor_data(ici_data.copy(), sensor_name="ICI")
    
    # ==========================================
    # 2. Process FLIR Sensor Data
    # ==========================================
    flir_clean = _process_sensor_data(flir_data.copy(), sensor_name="FLIR")
    
    # ==========================================
    # 3. Merge ICI and FLIR by SubjectID
    # ==========================================
    print("\n[clean] Merging ICI and FLIR data by SubjectID...")
    
    # Add sensor suffix to distinguish features
    ici_features = [c for c in ici_clean.columns if c not in ['SubjectID', 'Gender', 'Age', 'Ethnicity', 'OralTemp']]
    flir_features = [c for c in flir_clean.columns if c not in ['SubjectID', 'Gender', 'Age', 'Ethnicity', 'OralTemp']]
    
    ici_renamed = ici_clean.rename(columns={c: f"{c}_ici" for c in ici_features})
    flir_renamed = flir_clean.rename(columns={c: f"{c}_flir" for c in flir_features})
    
    # Merge on SubjectID, keeping demographics from ICI
    merged_df = pd.merge(
        ici_renamed,
        flir_renamed.drop(columns=['Gender', 'Age', 'Ethnicity', 'OralTemp'], errors='ignore'),
        on='SubjectID',
        how='inner'
    )
    
    print(f"[clean] Merged dataset shape: {merged_df.shape}")
    print(f"[clean] Subjects in both sensors: {len(merged_df)}")
    
    # ==========================================
    # 4. Final cleanup
    # ==========================================
    # Drop SubjectID (not a predictor)
    final_df = merged_df.drop(columns=['SubjectID'])
    
    # Verify no nulls in final dataset
    null_count = final_df.isnull().sum().sum()
    if null_count > 0:
        print(f"[clean] WARNING: {null_count} null values remain. Dropping affected rows...")
        final_df = final_df.dropna()
    
    print(f"\n[clean] FINAL DATASET:")
    print(f"  - Shape: {final_df.shape}")
    print(f"  - Features: {len(final_df.columns) - 1}")  # -1 for target
    print(f"  - Target: OralTemp (range: {final_df['OralTemp'].min():.2f} - {final_df['OralTemp'].max():.2f}°C)")
    print(f"  - Null values: {final_df.isnull().sum().sum()}")
    
    return final_df


def _process_sensor_data(df, sensor_name):
    """
    Process a single sensor's data:
    1. Drop spacer columns
    2. Average across rounds
    3. Create OralTemp target
    """
    print(f"\n[{sensor_name}] Processing {len(df)} subjects...")
    
    # ==========================================
    # Drop spacer columns (Unnamed:*)
    # ==========================================
    spacer_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    df = df.drop(columns=spacer_cols)
    print(f"[{sensor_name}] Dropped {len(spacer_cols)} spacer columns")
    
    # ==========================================
    # Identify column groups
    # ==========================================
    # Demographic columns (keep as-is)
    demo_cols = ['SubjectID', 'Gender', 'Age', 'Ethnicity']
    
    # Environmental columns (keep as-is)
    env_cols = ['T_atm', 'Humidity', 'Distance', 'Cosmetics', 'Time', 'Date']
    
    # Target columns
    target_cols = ['aveOralF', 'aveOralM']
    
    # Non-temperature columns to exclude
    exclude_cols = demo_cols + env_cols + target_cols
    
    # ==========================================
    # Identify round-based temperature columns
    # ==========================================
    # Two patterns exist in the data:
    # Pattern 1: ColumnName_1, ColumnName_2, ColumnName_3, ColumnName_4 (e.g., Max1R13_1)
    # Pattern 2: ColumnName1, ColumnName2, ColumnName3, ColumnName4 (e.g., T_RC1, T_offset1)
    
    temp_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Group columns by their base name
    # Pattern 1: ends with _1, _2, _3, _4
    # Pattern 2: ends with 1, 2, 3, 4 (but not preceded by underscore)
    
    base_name_map = {}  # base_name -> list of round columns
    
    for col in temp_cols:
        base_name = None
        
        # Check Pattern 1: ends with _1, _2, _3, _4
        match1 = re.match(r'^(.+)_([1-4])$', col)
        if match1:
            base_name = match1.group(1)
        else:
            # Check Pattern 2: ends with 1, 2, 3, 4 (no underscore before)
            match2 = re.match(r'^(.+[^_])([1-4])$', col)
            if match2:
                base_name = match2.group(1)
        
        if base_name:
            if base_name not in base_name_map:
                base_name_map[base_name] = []
            base_name_map[base_name].append(col)
    
    print(f"[{sensor_name}] Found {len(base_name_map)} temperature features to average across rounds...")
    
    # ==========================================
    # Average temperature columns across rounds
    # ==========================================
    averaged_data = {}
    for base_name, round_cols in base_name_map.items():
        if len(round_cols) >= 2:  # Only average if we have multiple rounds
            # Average across available rounds (handles NaN automatically)
            averaged_data[f"avg_{base_name}"] = df[round_cols].mean(axis=1)
    
    avg_df = pd.DataFrame(averaged_data)
    print(f"[{sensor_name}] Created {len(averaged_data)} averaged features")
    
    # ==========================================
    # Create OralTemp target based on Gender
    # ==========================================
    # Use aveOralF for females, aveOralM for males
    df['OralTemp'] = df.apply(
        lambda row: row['aveOralF'] if row['Gender'] == 'Female' else row['aveOralM'],
        axis=1
    )
    
    # ==========================================
    # Combine into clean dataframe
    # ==========================================
    # Keep: SubjectID, Demographics, Environment, Averaged temps, Target
    keep_demo = [c for c in demo_cols if c in df.columns]
    keep_env = [c for c in ['T_atm', 'Humidity', 'Distance'] if c in df.columns]  # Numeric env only
    
    clean_df = pd.concat([
        df[keep_demo].reset_index(drop=True),
        df[keep_env].reset_index(drop=True),
        avg_df.reset_index(drop=True),
        df[['OralTemp']].reset_index(drop=True)
    ], axis=1)
    
    print(f"[{sensor_name}] Cleaned shape: {clean_df.shape}")
    
    return clean_df
