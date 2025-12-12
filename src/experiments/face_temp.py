"""
Face Temperature Regression Experiment
- Predicts oral temperature from facial thermal measurements
- Uses ICI and FLIR sensor data combined
- Models: ElasticNet (baseline), XGBoost, Random Forest, Neural Network
- 3 splits (20/80, 50/50, 80/20) × 3 trials each
"""

import os
import sys
import json

# Add src directory to Python path so imports work from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True  # Prevent __pycache__ creation

import numpy as np
from sklearn.model_selection import train_test_split

from utils.load.load_face_temp import load_face_temp_data   
from utils.clean.clean_face_temp import clean_face_temp
from utils.eda.eda_face_temp import eda_face_temp

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment
from models.elastic_net import run_elastic_net_experiment

from graphs.face_temp_plots import (
    plot_face_temp_boosting_summary,
    plot_face_temp_random_forest_summary,
    plot_face_temp_neural_network_summary,
    plot_face_temp_elastic_net_summary,
    plot_face_temp_model_comparison,
)


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42
BOOSTING_NAME = "boosting"  # Now uses XGBoost under the hood
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"
# ELASTIC_NET_NAME = "elastic_net"


# ==========================================
# Model Generator Functions
# ==========================================
def generate_boosting(train_df, test_df, random_state, predictors, target_col):
    """
    Run XGBoost regression experiment.
    
    XGBoost has built-in regularization:
    - reg_alpha: L1 regularization (default 0)
    - reg_lambda: L2 regularization (default 1)
    """
    # Hyperparameter grid with regularization
    param_grid = {
        # Slightly larger ensemble plus a smaller learning rate
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.05],
        # Shallower trees regularize individual learners
        "model__max_depth": [1, 2, 3],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_alpha": [0, 0.1],      # L1 regularization
        "model__reg_lambda": [1, 10],      # L2 regularization
    }
    
    return run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
    )


def generate_random_forest(train_df, test_df, random_state, predictors, target_col):
    """Run Random Forest regression experiment."""
    
    # Hyperparameter grid for RF regression
    # Drop extremely shallow trees (depth 1–2) which underfit badly.
    param_grid = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [2, 5, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt"],
    }
    
    return run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
    )


def generate_neural_network(train_df, test_df, random_state, predictors, target_col):
    """Run Neural Network (MLP) regression experiment."""
    # For this dataset, large networks severely overfit and give unstable CV scores.
    # Use SMALLER, STRONGLY-REGULARIZED networks:
    #   - architectures around 20–32 units
    #   - stronger L2 (alpha)
    #   - smaller learning rate
    hidden_layer_sizes_grid = [
        (20,),
        (20, 10),
        (20, 20),
        (32,),
    ]

    param_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 0],       # much stronger L2 to cut overfitting
        "model__learning_rate_init": [0.001, 0.01],
        "model__batch_size": [32, 64, 150],
    }

    return run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="regression",
        random_state=random_state,
        hidden_layer_sizes_grid=hidden_layer_sizes_grid,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
    )


def generate_elastic_net(train_df, test_df, random_state, predictors, target_col):
    """
    Run ElasticNet as the LINEAR BASELINE.
    
    ElasticNet auto-finds best mix of L1/L2 regularization:
    - l1_ratio=0 → Pure Ridge (L2)
    - l1_ratio=1 → Pure Lasso (L1)
    - l1_ratio=0.5 → 50/50 mix
    
    If ElasticNet R² ≈ XGBoost R², the relationship is mostly linear.
    """
    param_grid = {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],  # Auto-find best L1/L2 mix
    }
    
    return run_elastic_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        random_state=random_state,
        param_grid=param_grid,
    )


# ==========================================
# Main Experiment
# ==========================================
def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ==========================================
    # 1. Load and Clean Data
    # ==========================================
    print("\n" + "="*60)
    print("FACE TEMPERATURE REGRESSION EXPERIMENT")
    print("="*60)
    
    ici_data, flir_data = load_face_temp_data(curr_dir)
    clean_df = clean_face_temp(ici_data, flir_data)
    
    # ==========================================
    # 2. Run EDA
    # ==========================================
    eda_face_temp(clean_df)
    
    # ==========================================
    # 3. Prepare Features and Target
    # ==========================================
    target_col = "OralTemp"
    
    # Separate predictors (numeric only for now, encode categoricals later in pipeline)
    cat_cols = clean_df.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in clean_df.columns if c != target_col and c not in cat_cols]
    predictors = num_cols + cat_cols
    
    print(f"\n[face_temp.py] Target: {target_col}")
    print(f"[face_temp.py] Predictors: {len(predictors)} features")
    print(f"[face_temp.py] Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")
    
    # ==========================================
    # 4. Experiment Configuration
    # ==========================================
    split_configs = {
        "20_80": 0.80,  # 20% train, 80% test
        "50_50": 0.50,  # 50% train, 50% test
        "80_20": 0.20,  # 80% train, 20% test
    }
    
    n_trials = 3
    
    results = {
        # ELASTIC_NET_NAME: {},  # Linear baseline (auto-finds best L1/L2 mix)
        BOOSTING_NAME: {},     # XGBoost with built-in regularization
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
        
        # results[ELASTIC_NET_NAME][split_name] = []
        results[BOOSTING_NAME][split_name] = []
        results[RANDOM_FOREST_NAME][split_name] = []
        results[NEURAL_NETWORK_NAME][split_name] = []
        
        for trial in range(n_trials):
            print(f"\n----- Trial {trial + 1}/{n_trials} -----")
            
            # Different random_state per trial
            rs = RANDOM_STATE + trial
            
            # For regression, we can't stratify on continuous target
            # Use random split (or bin target for stratification if needed)
            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_size,
                random_state=rs,
                shuffle=True,
            )
            
            print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
            
            # ----- Run ElasticNet (Linear Baseline) -----
            # print(f"\n>>> Running ElasticNet Baseline (Trial {trial + 1})...")
            # elastic_result = generate_elastic_net(train_df, test_df, rs, predictors, target_col)
            
            # elastic_record = {
            #     "trial": trial,
            #     "best_params": elastic_result["best_params"],
            #     "cv_train_score": elastic_result["cv_train_score"],
            #     "cv_val_score": elastic_result["cv_val_score"],
            #     "cv_scoring": elastic_result["cv_scoring"],
            #     "regularization_type": elastic_result.get("regularization_type"),
            #     "n_nonzero_coefs": elastic_result.get("n_nonzero_coefs"),
            #     "test_metrics": {
            #         "mse": elastic_result["test_metrics"]["mse"],
            #         "rmse": elastic_result["test_metrics"]["rmse"],
            #         "mae": elastic_result["test_metrics"]["mae"],
            #         "r2": elastic_result["test_metrics"]["r2"],
            #     },
            #     "y_test": elastic_result["y_test"].tolist(),
            #     "y_pred": elastic_result["y_pred"].tolist(),
            #     "coefficients": (
            #         elastic_result["coefficients"].tolist()
            #         if elastic_result.get("coefficients") is not None
            #         else None
            #     ),
            #     "feature_names": elastic_result.get("feature_names"),
            # }
            # results[ELASTIC_NET_NAME][split_name].append(elastic_record)
            
            # ----- Run Boosting (XGBoost) -----
            print(f"\n>>> Running XGBoost (Trial {trial + 1})...")
            boosting_result = generate_boosting(train_df, test_df, rs, predictors, target_col)
            
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
            
            # ----- Run Random Forest -----
            print(f"\n>>> Running Random Forest (Trial {trial + 1})...")
            rf_result = generate_random_forest(train_df, test_df, rs, predictors, target_col)
            
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
            
            # ----- Run Neural Network -----
            print(f"\n>>> Running Neural Network (Trial {trial + 1})...")
            nn_result = generate_neural_network(train_df, test_df, rs, predictors, target_col)
            
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
    
    out_path = os.path.join(results_dir, "face_temp_all_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[face_temp.py] Saved all results to {out_path}")
    
    # ==========================================
    # 7. Generate Plots
    # ==========================================
    plots_base = os.path.join(curr_dir, "../..", "plots/face_temp_plots/results")
    
    # ElasticNet baseline plots
    # elastic_plots_dir = os.path.join(plots_base, "elastic_net")
    # plot_face_temp_elastic_net_summary(results[ELASTIC_NET_NAME], elastic_plots_dir)
    
    # Boosting (XGBoost) plots
    boosting_plots_dir = os.path.join(plots_base, "boosting")
    plot_face_temp_boosting_summary(results[BOOSTING_NAME], boosting_plots_dir)
    
    rf_plots_dir = os.path.join(plots_base, "random_forest")
    plot_face_temp_random_forest_summary(results[RANDOM_FOREST_NAME], rf_plots_dir)
    
    nn_plots_dir = os.path.join(plots_base, "neural_network")
    plot_face_temp_neural_network_summary(results[NEURAL_NETWORK_NAME], nn_plots_dir)
    
    # Comparison plot across all models
    comparison_dir = os.path.join(plots_base, "comparison")
    plot_face_temp_model_comparison(results, comparison_dir)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
