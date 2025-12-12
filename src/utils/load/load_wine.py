import os
import pandas as pd


def load_wine_data(curr_dir: str) -> pd.DataFrame:
    """
    Load the UCI Wine dataset.

    The raw file `wine.data` has:
      - First column: class label (1, 2, 3)
      - Next 13 columns: continuous features (see wine.names for details)

    We expose a tidy DataFrame with column names:
      - 'Class' (target)
      - 13 feature columns
    """
    data_path = os.path.join(curr_dir, "../../datasets/WineSets/wine.data")

    col_names = [
        "Class",            # target (1, 2, 3)
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]

    df = pd.read_csv(data_path, header=None, names=col_names)

    print(f"[load_wine] Loaded wine dataset from {data_path}")
    print(f"[load_wine] Shape: {df.shape}")
    print(f"[load_wine] Columns: {df.columns.tolist()}")

    return df


