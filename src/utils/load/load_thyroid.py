import os
import pandas as pd


def load_thyroid_data(curr_dir: str) -> pd.DataFrame:
    """
    Load the Thyroid Cancer dataset.

    Source file: `datasets/ThyroidCancerSets/Thyroid_Diff.csv`

    The CSV already contains a header row. We simply read it into a DataFrame
    and report basic shape/column information for sanity checking.
    """
    data_path = os.path.join(curr_dir, "../../datasets/ThyroidCancerSets/Thyroid_Diff.csv")

    df = pd.read_csv(data_path)

    print(f"[load_thyroid] Loaded thyroid cancer dataset from {data_path}")
    print(f"[load_thyroid] Shape: {df.shape}")
    print(f"[load_thyroid] Columns: {df.columns.tolist()}")

    return df


