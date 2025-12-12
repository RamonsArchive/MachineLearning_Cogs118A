import pandas as pd


def clean_wine(data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for the Wine dataset.

    - Confirms there are no missing values (drops rows with nulls if any).
    - Leaves all 13 chemical analysis features as-is.
    - Keeps 'Class' as the target column (1, 2, 3).
    """
    df = data.copy()

    print("\n" + "=" * 60)
    print("CLEANING WINE DATASET")
    print("=" * 60)

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    if total_nulls == 0:
        print("[clean_wine] ✓ No missing values detected.")
    else:
        print(f"[clean_wine] ⚠ Found {total_nulls} missing values. Dropping rows with nulls.")
        print(null_counts[null_counts > 0])
        df = df.dropna()
        print(f"[clean_wine] New shape after dropping nulls: {df.shape}")

    print(f"[clean_wine] Final shape: {df.shape}")
    print(f"[clean_wine] Target column: 'Class'")

    return df


