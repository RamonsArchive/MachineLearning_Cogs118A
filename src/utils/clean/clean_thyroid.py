import pandas as pd


def clean_thyroid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the Thyroid Cancer dataset.

    - Drops rows with missing values (if any).
    - Leaves categorical columns as strings so downstream pipelines can one-hot
      encode them.
    - Uses the 'Recurred' column as the classification target.
      (Predict whether the cancer recurred: 'Yes' / 'No'.)
    - Drops no other columns at this stage; feature selection happens in the
      experiment script.
    """
    df = data.copy()

    print("\n" + "=" * 60)
    print("CLEANING THYROID CANCER DATASET")
    print("=" * 60)

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    if total_nulls == 0:
        print("[clean_thyroid] ✓ No missing values detected.")
    else:
        print(f"[clean_thyroid] ⚠ Found {total_nulls} missing values. Dropping rows with nulls.")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print(f"[clean_thyroid] New shape after dropping nulls: {df.shape}")

    if "Recurred" not in df.columns:
        raise ValueError("[clean_thyroid] Expected 'Recurred' column to exist as target.")

    print(f"[clean_thyroid] Final shape: {df.shape}")
    print(f"[clean_thyroid] Target column: 'Recurred'")

    return df


