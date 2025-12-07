import os
import sys
import json

# Prevent Python from caching bytecode (.pyc files)
sys.dont_write_bytecode = True

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.load.load_bank import load_bank_data
from utils.clean.clean_bank import clean_bank
from utils.eda.eda_bank import eda_bank
from sklearn.model_selection import train_test_split

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment
from models.svm import run_svm_experiment

from graphs.bank_boosting_plots import plot_bank_boosting_summary
from graphs.bank_random_forest_plots import plot_bank_random_forest_summary
from graphs.bank_neural_network_plots import plot_bank_neural_network_summary
from graphs.bank_svm_plots import plot_bank_svm_summary



RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"
SVM_NAME = "svm"

def generate_boosting(train_df, test_df, RANDOM_STATE):
    target_col = "y"
    predictors = [c for c in train_df.columns if c != target_col]
    param_grid = {
        "model__n_estimators": [300, 600, 1000, 1200, 2000],
        "model__learning_rate": [0.005, 0.01, 0.03, 0.1],
        "model__max_depth": [1, 2, 3, 5, 7, 9],
    }
    # param_grid = {
    #     "model__n_estimators": [1200],
    #     "model__learning_rate": [0.01],
    #     "model__max_depth": [2],
    # }

    results = run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        param_grid=param_grid,
    )

    return results

def generate_random_forest(train_df, test_df, RANDOM_STATE):
    target_col = "y"
    predictors = [c for c in train_df.columns if c != target_col]
    
    # Reduced grid for efficiency (72 combos vs 288)
    # Random Forest is robust; fewer hyperparams often sufficient
    param_grid = {
        "model__n_estimators": [200, 500, 1000],    # fewer trees still effective
        "model__max_depth": [10, 20, None],         # 3 depths
        "model__min_samples_split": [2, 5],         # 2 values
        "model__min_samples_leaf": [1, 2],          # 2 values
        "model__max_features": ['sqrt', 'log2'],    # 2 feature selection methods
    }

    # param_grid = {
    #     "model__n_estimators": [1000],    # fewer trees still effective
    #     "model__max_depth": [10],         # 3 depths
    #     "model__min_samples_split": [2],         # 2 values
    #     "model__min_samples_leaf": [1],          # 2 values
    #     "model__max_features": ['sqrt'],    # 2 feature selection methods
    # }


    results = run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        param_grid=param_grid,
    )

    return results

def generate_neural_network(train_df, test_df, RANDOM_STATE):
    target_col = "y"
    predictors = [c for c in train_df.columns if c != target_col]

    # Network architectures to try
    hidden_layer_sizes_grid = [
        (100,),           # 1 layer
        (100, 50),        # 2 layers
        (100, 50, 25),    # 3 layers
    ]
    
    # Other hyperparameters (reduced for efficiency)
    # Full grid: 3 * 3 * 3 * 3 = 81 combos
    param_grid = {
        "model__learning_rate_init": [0.001, 0.01, 0.03],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__batch_size": [32, 64, 'auto'],
    }

    results = run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        hidden_layer_sizes_grid=hidden_layer_sizes_grid,
        param_grid=param_grid,
    )
    
    return results

def generate_svm(train_df, test_df, RANDOM_STATE):
    target_col = "y"
    predictors = [c for c in train_df.columns if c != target_col]
    
    # SVM Hyperparameters:
    # - kernel: 'rbf' (flexible, non-linear) vs 'linear' (faster, good for high-dim)
    # - C: Regularization (higher = tighter fit, risk overfitting)
    # - gamma: Kernel coefficient (higher = complex boundaries)
    # Grid: 2 * 4 * 3 = 24 combos (reasonable)
    param_grid = {
        "model__kernel": ['rbf', 'linear'],
        "model__C": [1, 10, 100],
        "model__gamma": ['scale', 0.1],  # ignored for linear kernel
    }

    results = run_svm_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=RANDOM_STATE,
        param_grid=param_grid,
    )

    return results



def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_bank_data(curr_dir)
    clean_df = clean_bank(data)
    
    # DEBUG: Verify data leakage columns are removed
    print("\n=== DEBUG: Checking for data leakage ===")
    print(f"Columns in clean_df: {clean_df.columns.tolist()}")
    print(f"'duration' in columns: {'duration' in clean_df.columns}")
    print(f"'campaign' in columns: {'campaign' in clean_df.columns}")
    if 'duration' in clean_df.columns or 'campaign' in clean_df.columns:
        raise ValueError("DATA LEAKAGE DETECTED! duration or campaign still in data!")
    print("=== No data leakage detected ===\n")
    
    eda_bank(clean_df)  # ok if this just prints / sanity checks

    # ------------------------------------------------------------------
    # RESULTS STRUCTURE:
    # results = {
    #   "bank": {
    #       "boosting": {
    #           "20_80": [ {trial 0 metrics}, {trial 1 metrics}, {trial 2 metrics} ],
    #           "50_50": [ ... ],
    #           "80_20": [ ... ]
    #       },
    #       "random_forest": { ... },
    #       ...
    #   },
    #   "other_dataset": { ... }
    # }
    # ------------------------------------------------------------------
    results = {
        BOOSTING_NAME: {},
        RANDOM_FOREST_NAME: {},
        NEURAL_NETWORK_NAME: {},
        SVM_NAME: {},
    }

    # Train/test split configs (train/test)
    split_configs = {
        "20_80": 0.80,   # train=20%, test=80%  -> test_size = 0.80
        "50_50": 0.50,   # train=50%, test=50%  -> test_size = 0.50
        "80_20": 0.20,   # train=80%, test=20%  -> test_size = 0.20
    }

    n_trials = 3

    for split_name, test_size in split_configs.items():
        print(f"\n===== SPLIT {split_name} (test_size={test_size}) =====")

        results[BOOSTING_NAME][split_name] = []
        results[RANDOM_FOREST_NAME][split_name] = []
        results[NEURAL_NETWORK_NAME][split_name] = []
        results[SVM_NAME][split_name] = []

        for trial in range(n_trials):
            # Different random_state per trial so splits differ
            rs = RANDOM_STATE + trial

            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_size,
                random_state=rs,
                shuffle=True,
                stratify=clean_df["y"]  # stratify for classification
            )

            # ----- Run Models -----
            boosting_result = generate_boosting(train_df, test_df, rs)
            random_forest_result = generate_random_forest(train_df, test_df, rs)
            neural_network_result = generate_neural_network(train_df, test_df, rs)
            svm_result = generate_svm(train_df, test_df, rs)

            # EXPECTED fields in boosting_result (you can tweak names to your impl):
            # {
            #   "best_params": {...},
            #   "cv_best_score": float,
            #   "train_accuracy": float,
            #   "val_accuracy": float,
            #   "test_accuracy": float,
            #   ... (other metrics ok)
            # }

            boosting_trial_record = {
                "trial": trial,
                "best_params": boosting_result["best_params"],
                "cv_train_score": boosting_result["cv_train_score"],
                "cv_val_score": boosting_result["cv_val_score"],
                "test_accuracy": boosting_result["test_metrics"]["accuracy"],
                "test_metrics": boosting_result["test_metrics"],
                # For plots / confusion / ROC:
                "y_test": boosting_result["y_test"].tolist(),
                "y_pred": boosting_result["y_pred"].tolist(),
                "y_proba": (
                    boosting_result["y_proba"].tolist()
                    if boosting_result["y_proba"] is not None
                    else None
                ),
            }

            random_forest_record = {
                "trial": trial,
                "best_params": random_forest_result["best_params"],
                "cv_train_score": random_forest_result["cv_train_score"],
                "cv_val_score": random_forest_result["cv_val_score"],
                "test_accuracy": random_forest_result["test_metrics"]["accuracy"],
                "test_metrics": random_forest_result["test_metrics"],
                # For plots / confusion / ROC:
                "y_test": random_forest_result["y_test"].tolist(),
                "y_pred": random_forest_result["y_pred"].tolist(),
                "y_proba": (
                    random_forest_result["y_proba"].tolist()
                    if random_forest_result["y_proba"] is not None
                    else None
                ),
                # Feature importances (unique to RF)
                "feature_importances": (
                    random_forest_result["feature_importances"].tolist()
                    if random_forest_result["feature_importances"] is not None
                    else None
                ),
                "feature_names": random_forest_result["feature_names"],
            }

            neural_network_record = {
                "trial": trial,
                "best_params": neural_network_result["best_params"],
                "cv_train_score": neural_network_result["cv_train_score"],
                "cv_val_score": neural_network_result["cv_val_score"],
                "test_accuracy": neural_network_result["test_metrics"]["accuracy"],
                "test_metrics": neural_network_result["test_metrics"],
                # For plots / confusion / ROC:
                "y_test": neural_network_result["y_test"].tolist(),
                "y_pred": neural_network_result["y_pred"].tolist(),
                "y_proba": (
                    neural_network_result["y_proba"].tolist()
                    if neural_network_result["y_proba"] is not None
                    else None
                ),
                # Training info (unique to NN)
                "training_info": neural_network_result["training_info"],
            }

            svm_record = {
                "trial": trial,
                "best_params": svm_result["best_params"],
                "cv_train_score": svm_result["cv_train_score"],
                "cv_val_score": svm_result["cv_val_score"],
                "test_accuracy": svm_result["test_metrics"]["accuracy"],
                "test_metrics": svm_result["test_metrics"],
                # For plots / confusion / ROC:
                "y_test": svm_result["y_test"].tolist(),
                "y_pred": svm_result["y_pred"].tolist(),
                "y_proba": (
                    svm_result["y_proba"].tolist()
                    if svm_result["y_proba"] is not None
                    else None
                ),
                # SVM info (unique to SVM)
                "svm_info": svm_result["svm_info"],
            }

            results[BOOSTING_NAME][split_name].append(boosting_trial_record)
            results[RANDOM_FOREST_NAME][split_name].append(random_forest_record)
            results[NEURAL_NETWORK_NAME][split_name].append(neural_network_record)
            results[SVM_NAME][split_name].append(svm_record)

    # Save results to JSON
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all results together
    out_path = os.path.join(results_dir, "bank_all_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bank.py] Saved all results to {out_path}")

    # # Plot Boosting results
    boosting_plots_dir = os.path.join(curr_dir, "../..", "plots/bank_plots", "bank_boosting_plots")
    plot_bank_boosting_summary(results[BOOSTING_NAME], boosting_plots_dir)
    
    # Plot Random Forest results
    rf_plots_dir = os.path.join(curr_dir, "../..", "plots/bank_plots", "bank_random_forest_plots")
    plot_bank_random_forest_summary(results[RANDOM_FOREST_NAME], rf_plots_dir)
    
    # Plot Neural Network results
    nn_plots_dir = os.path.join(curr_dir, "../..", "plots/bank_plots", "bank_neural_network_plots")
    plot_bank_neural_network_summary(results[NEURAL_NETWORK_NAME], nn_plots_dir)
    
    # Plot SVM results
    svm_plots_dir = os.path.join(curr_dir, "../..", "plots/bank_plots", "bank_svm_plots")
    plot_bank_svm_summary(results[SVM_NAME], svm_plots_dir)


if __name__ == "__main__":
    main()