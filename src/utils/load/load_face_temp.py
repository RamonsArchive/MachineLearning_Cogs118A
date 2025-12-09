import pandas as pd
import os


def load_face_temp_data(curr_dir):
    """
    Load face temperature data from ICI and FLIR sensors.
    The CSV files have 3 header rows - we use row 3 (index 2) as column names.
    """
    # Skip first 2 rows (descriptive headers), use row 3 as column names
    ici_path = os.path.join(curr_dir, "../../datasets/FaceTempSets/ICI_groups1and2.csv")
    flir_path = os.path.join(curr_dir, "../../datasets/FaceTempSets/FLIR_groups1and2.csv")
    
    # Row 0: "Round 1:", "Round 2:", etc.
    # Row 1: Column descriptions 
    # Row 2: Actual column names (SubjectID, T_offset1, etc.)
    ici_data = pd.read_csv(ici_path, header=2)
    flir_data = pd.read_csv(flir_path, header=2)
    
    print(f"[load_face_temp] Loaded ICI data: {ici_data.shape}")
    print(f"[load_face_temp] Loaded FLIR data: {flir_data.shape}")
    
    return ici_data, flir_data
