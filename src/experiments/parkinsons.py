"""
Parkinson's Telemonitoring Regression Experiment
- Predicts total_UPDRS score (disease severity) from voice measurements
- 5,875 recordings from 42 patients over 6 months
- Models: ElasticNet (baseline), XGBoost, Random Forest, Neural Network
- 3 splits (20/80, 50/50, 80/20) × 3 trials each

TARGET: total_UPDRS (not motor_UPDRS)
- total_UPDRS is the standard research outcome measure
- motor_UPDRS is a COMPONENT of total_UPDRS (using both = data leakage)
- We drop motor_UPDRS from predictors entirely
"""

import os
import sys
import json

# Add src directory to Python path so imports work from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True

import numpy as np
from sklearn.model_selection import train_test_split

from utils.load.load_parkinsons import load_parkinsons_data
from utils.clean.clean_parkinsons import clean_parkinsons
from utils.eda.eda_parkinsons import eda_parkinsons

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment
from models.elastic_net import run_elastic_net_experiment

from graphs.parkinsons_plots import (
    plot_parkinsons_boosting_summary,
    plot_parkinsons_random_forest_summary,
    plot_parkinsons_neural_network_summary,
    plot_parkinsons_elastic_net_summary,
    plot_parkinsons_model_comparison,
)


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42
ELASTIC_NET_NAME = "elastic_net"
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"

TARGET_COL = "total_UPDRS"  # Standard research outcome measure


# ==========================================
# Model Generator Functions
# ==========================================
def generate_elastic_net(train_df, test_df, random_state, predictors, target_col):
    """
    ElasticNet - Linear baseline.
    Auto-finds best L1/L2 mix.
    """
    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    return run_elastic_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        random_state=random_state,
        param_grid=param_grid,
    )


def generate_boosting(train_df, test_df, random_state, predictors, target_col):
    """
    XGBoost regression.
    Grid tuned for ~6k samples, 19 features, weak correlations.
    """
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__reg_alpha": [0, 0.1],
        "model__reg_lambda": [1, 10],
    }
    return run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        param_grid=param_grid,
    )


def generate_random_forest(train_df, test_df, random_state, predictors, target_col):
    """
    Random Forest regression.
    Grid tuned for ~6k samples, 19 features.
    """
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5],
    }
    return run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        param_grid=param_grid,
    )


def generate_neural_network(train_df, test_df, random_state, predictors, target_col):
    """
    Neural Network (MLP) regression.
    Smaller networks for ~6k samples to avoid overfitting.
    """
    param_grid = {
        "model__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
        "model__alpha": [0.001, 0.01, 0.1],
        "model__learning_rate_init": [0.001, 0.01],
        "model__batch_size": [32, 64],
    }
    return run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        param_grid=param_grid,
    )


# ==========================================
# Main Experiment
# ==========================================
def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*60)
    print("PARKINSON'S TELEMONITORING REGRESSION EXPERIMENT")
    print("="*60)
    
    # ==========================================
    # 1. Load and Clean Data
    # ==========================================
    data = load_parkinsons_data(curr_dir)
    clean_df = clean_parkinsons(data)
    
    # ==========================================
    # 2. Run EDA
    # ==========================================
    eda_parkinsons(clean_df)
    
    # ==========================================
    # 3. Define Features and Target
    # ==========================================
    # IMPORTANT: Drop both subject# (ID) and motor_UPDRS (component of total_UPDRS)
    # Using motor_UPDRS as predictor would be DATA LEAKAGE!
    exclude_cols = ['subject#', 'motor_UPDRS', TARGET_COL]
    predictors = [c for c in clean_df.columns if c not in exclude_cols]
    
    print(f"\n[parkinsons.py] Target: {TARGET_COL}")
    print(f"[parkinsons.py] Predictors: {len(predictors)} features")
    print(f"[parkinsons.py] Excluded: subject# (ID), motor_UPDRS (data leakage)")
    
    # ==========================================
    # 4. Experiment Configuration
    # ==========================================
    split_configs = {
        "20_80": 0.80,
        "50_50": 0.50,
        "80_20": 0.20,
    }
    
    n_trials = 3
    
    results = {
        ELASTIC_NET_NAME: {},
        BOOSTING_NAME: {},
        RANDOM_FOREST_NAME: {},
        NEURAL_NETWORK_NAME: {},
    }
    
    # ==========================================
    # 5. Run Experiments
    # ==========================================
    for split_name, test_size in split_configs.items():
        print(f"\n{'='*60}")
        print(f"SPLIT {split_name} (test_size={test_size})")
        print(f"{'='*60}")
        
        results[ELASTIC_NET_NAME][split_name] = []
        results[BOOSTING_NAME][split_name] = []
        results[RANDOM_FOREST_NAME][split_name] = []
        results[NEURAL_NETWORK_NAME][split_name] = []
        
        for trial in range(n_trials):
            print(f"\n----- Trial {trial + 1}/{n_trials} -----")
            
            rs = RANDOM_STATE + trial
            
            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_size,
                random_state=rs,
                shuffle=True,
            )
            
            print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
            
            # ----- ElasticNet (Linear Baseline) -----
            print(f"\n>>> Running ElasticNet Baseline (Trial {trial + 1})...")
            elastic_result = generate_elastic_net(train_df, test_df, rs, predictors, TARGET_COL)
            
            elastic_record = {
                "trial": trial,
                "best_params": elastic_result["best_params"],
                "cv_train_score": elastic_result["cv_train_score"],
                "cv_val_score": elastic_result["cv_val_score"],
                "cv_scoring": elastic_result["cv_scoring"],
                "regularization_type": elastic_result.get("regularization_type"),
                "test_metrics": {
                    "mse": elastic_result["test_metrics"]["mse"],
                    "rmse": elastic_result["test_metrics"]["rmse"],
                    "mae": elastic_result["test_metrics"]["mae"],
                    "r2": elastic_result["test_metrics"]["r2"],
                },
                "y_test": elastic_result["y_test"].tolist(),
                "y_pred": elastic_result["y_pred"].tolist(),
                "coefficients": (
                    elastic_result["coefficients"].tolist()
                    if elastic_result.get("coefficients") is not None
                    else None
                ),
                "feature_names": elastic_result.get("feature_names"),
            }
            results[ELASTIC_NET_NAME][split_name].append(elastic_record)
            
            # ----- XGBoost -----
            print(f"\n>>> Running XGBoost (Trial {trial + 1})...")
            boosting_result = generate_boosting(train_df, test_df, rs, predictors, TARGET_COL)
            
            boosting_record = {
                "trial": trial,
                "best_params": boosting_result["best_params"],
                "cv_train_score": boosting_result["cv_train_score"],
                "cv_val_score": boosting_result["cv_val_score"],
                "cv_scoring": boosting_result["cv_scoring"],
                "test_metrics": {
                    "mse": boosting_result["test_metrics"]["mse"],
                    "rmse": boosting_result["test_metrics"]["rmse"],
                    "mae": boosting_result["test_metrics"]["mae"],
                    "r2": boosting_result["test_metrics"]["r2"],
                },
                "y_test": boosting_result["y_test"].tolist(),
                "y_pred": boosting_result["y_pred"].tolist(),
                "feature_importances": (
                    boosting_result["feature_importances"].tolist()
                    if boosting_result.get("feature_importances") is not None
                    else None
                ),
                "feature_names": boosting_result.get("feature_names"),
            }
            results[BOOSTING_NAME][split_name].append(boosting_record)
            
            # ----- Random Forest -----
            print(f"\n>>> Running Random Forest (Trial {trial + 1})...")
            rf_result = generate_random_forest(train_df, test_df, rs, predictors, TARGET_COL)
            
            rf_record = {
                "trial": trial,
                "best_params": rf_result["best_params"],
                "cv_train_score": rf_result["cv_train_score"],
                "cv_val_score": rf_result["cv_val_score"],
                "cv_scoring": rf_result["cv_scoring"],
                "test_metrics": {
                    "mse": rf_result["test_metrics"]["mse"],
                    "rmse": rf_result["test_metrics"]["rmse"],
                    "mae": rf_result["test_metrics"]["mae"],
                    "r2": rf_result["test_metrics"]["r2"],
                },
                "y_test": rf_result["y_test"].tolist(),
                "y_pred": rf_result["y_pred"].tolist(),
                "feature_importances": (
                    rf_result["feature_importances"].tolist()
                    if rf_result.get("feature_importances") is not None
                    else None
                ),
                "feature_names": rf_result.get("feature_names"),
            }
            results[RANDOM_FOREST_NAME][split_name].append(rf_record)
            
            # ----- Neural Network -----
            print(f"\n>>> Running Neural Network (Trial {trial + 1})...")
            nn_result = generate_neural_network(train_df, test_df, rs, predictors, TARGET_COL)
            
            nn_record = {
                "trial": trial,
                "best_params": nn_result["best_params"],
                "cv_train_score": nn_result["cv_train_score"],
                "cv_val_score": nn_result["cv_val_score"],
                "cv_scoring": nn_result["cv_scoring"],
                "test_metrics": {
                    "mse": nn_result["test_metrics"]["mse"],
                    "rmse": nn_result["test_metrics"]["rmse"],
                    "mae": nn_result["test_metrics"]["mae"],
                    "r2": nn_result["test_metrics"]["r2"],
                },
                "y_test": nn_result["y_test"].tolist(),
                "y_pred": nn_result["y_pred"].tolist(),
                "training_info": nn_result.get("training_info"),
            }
            results[NEURAL_NETWORK_NAME][split_name].append(nn_record)
    
    # ==========================================
    # 6. Save Results
    # ==========================================
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_path = os.path.join(results_dir, "parkinsons_all_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[parkinsons.py] Saved all results to {out_path}")
    
    # ==========================================
    # 7. Generate Plots
    # ==========================================
    plots_base = os.path.join(curr_dir, "../..", "plots/parkinsons_plots/results")
    
    elastic_plots_dir = os.path.join(plots_base, "elastic_net")
    plot_parkinsons_elastic_net_summary(results[ELASTIC_NET_NAME], elastic_plots_dir)
    
    boosting_plots_dir = os.path.join(plots_base, "boosting")
    plot_parkinsons_boosting_summary(results[BOOSTING_NAME], boosting_plots_dir)
    
    rf_plots_dir = os.path.join(plots_base, "random_forest")
    plot_parkinsons_random_forest_summary(results[RANDOM_FOREST_NAME], rf_plots_dir)
    
    nn_plots_dir = os.path.join(plots_base, "neural_network")
    plot_parkinsons_neural_network_summary(results[NEURAL_NETWORK_NAME], nn_plots_dir)
    
    comparison_dir = os.path.join(plots_base, "comparison")
    plot_parkinsons_model_comparison(results, comparison_dir)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
