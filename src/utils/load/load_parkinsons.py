import pandas as pd
import os

"""
Parkinson's Telemonitoring Dataset
==================================

5,875 voice recordings from 42 people with early-stage Parkinson's disease.
Recorded at home over 6 months using a telemonitoring device.

TASK: REGRESSION - Predict UPDRS scores from voice measurements

TARGETS:
- motor_UPDRS: Motor symptoms score (0-108, higher = worse symptoms)
- total_UPDRS: Total symptoms score (0-176, higher = worse symptoms)

UPDRS = Unified Parkinson's Disease Rating Scale (clinical assessment)

VOICE FEATURES (16 measures):
- Jitter: Variation in fundamental frequency (pitch instability)
  - Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP
  - Higher jitter = more pitch instability = worse voice control

- Shimmer: Variation in amplitude (loudness instability)
  - Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA
  - Higher shimmer = more loudness variation = worse voice control

- NHR: Noise-to-Harmonics Ratio (more noise = worse voice quality)
- HNR: Harmonics-to-Noise Ratio (more harmonics = better voice quality)
- RPDE: Recurrence Period Density Entropy (nonlinear complexity)
- DFA: Detrended Fluctuation Analysis (fractal scaling)
- PPE: Pitch Period Entropy (pitch variation measure)

WHY VOICE? Parkinson's affects motor control, including vocal cord muscles.
Voice changes often appear BEFORE other symptoms, making it useful for early detection.
"""


def load_parkinsons_data(curr_dir):
    """
    Load Parkinson's telemonitoring dataset.
    
    Returns:
        pd.DataFrame: The full dataset with all columns
        
    Note: The .names file is just documentation, not data to load.
    """
    data_path = os.path.join(curr_dir, "../../datasets/ParkinsonsTelemonitoringSets/parkinsons_updrs.data")
    data = pd.read_csv(data_path)
    
    print(f"[load_parkinsons] Loaded {len(data)} voice recordings from {data['subject#'].nunique()} patients")
    print(f"[load_parkinsons] Columns: {list(data.columns)}")
    
    return data
