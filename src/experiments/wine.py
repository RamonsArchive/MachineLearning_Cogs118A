import os
import sys
import json

# Prevent Python from caching bytecode (.pyc files)
sys.dont_write_bytecode = True

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.model_selection import train_test_split

from utils.load.load_wine import load_wine_data
from utils.clean.clean_wine import clean_wine
from utils.eda.eda_wine import eda_wine

from models.boosting import run_boosting_experiment
from models.random_forest import run_random_forest_experiment
from models.neural_net import run_neural_net_experiment

from graphs.wine_boosting_plots import plot_wine_boosting_summary
from graphs.wine_random_forest_plots import plot_wine_random_forest_summary
from graphs.wine_neural_network_plots import plot_wine_neural_network_summary
from graphs.wine_model_comparison import plot_wine_model_comparison


RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"


def generate_boosting(train_df, test_df, random_state):
    """
    XGBoost classification on the Wine dataset (3 classes).
    Use a SMALLER grid than bank because the dataset is tiny (178 rows).
    """
    target_col = "Class"
    predictors = [c for c in train_df.columns if c != target_col]

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0],
        "model__reg_alpha": [0, 0.1],
        "model__reg_lambda": [1, 10],
    }

    results = run_boosting_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        param_grid=param_grid,
    )

    return results


def generate_random_forest(train_df, test_df, random_state):
    """Random Forest classification with a compact grid for the small Wine dataset."""
    target_col = "Class"
    predictors = [c for c in train_df.columns if c != target_col]

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, None],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }

    results = run_random_forest_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        param_grid=param_grid,
    )

    return results


def generate_neural_network(train_df, test_df, random_state):
    """
    Neural Network (MLP) classification.
    Use very small networks to avoid overfitting 178 samples.
    """
    target_col = "Class"
    predictors = [c for c in train_df.columns if c != target_col]

    hidden_layer_sizes_grid = [
        (16,),
        (32,),
        (16, 8),
    ]

    param_grid = {
        "model__learning_rate_init": [0.001, 0.01],
        "model__alpha": [0.0001, 0.001],
        "model__batch_size": [16, 32],
    }

    results = run_neural_net_experiment(
        train_df=train_df,
        test_df=test_df,
        predictors=predictors,
        target_col=target_col,
        problem_type="classification",
        random_state=random_state,
        hidden_layer_sizes_grid=hidden_layer_sizes_grid,
        param_grid=param_grid,
    )

    return results


def main():
    print("\n" + "=" * 60)
    print("WINE QUALITY CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load and clean
    data = load_wine_data(curr_dir)
    clean_df = clean_wine(data)

    # 2. EDA (includes naive baseline)
    eda_wine(clean_df)

    # 3. Experiment configuration
    split_configs = {
        "20_80": 0.80,
        "50_50": 0.50,
        "80_20": 0.20,
    }
    n_trials = 3

    results = {
        BOOSTING_NAME: {},
        RANDOM_FOREST_NAME: {},
        NEURAL_NETWORK_NAME: {},
    }

    # 4. Run experiments
    for split_name, test_size in split_configs.items():
        print(f"\n===== SPLIT {split_name} (test_size={test_size}) =====")

        results[BOOSTING_NAME][split_name] = []
        results[RANDOM_FOREST_NAME][split_name] = []
        results[NEURAL_NETWORK_NAME][split_name] = []

        for trial in range(n_trials):
            rs = RANDOM_STATE + trial

            train_df, test_df = train_test_split(
                clean_df,
                test_size=test_size,
                random_state=rs,
                shuffle=True,
                stratify=clean_df["Class"],
            )

            print(f"Trial {trial + 1}/{n_trials} – Train size: {len(train_df)}, Test size: {len(test_df)}")

            # Boosting
            print(f"\n>>> Running XGBoost (Trial {trial + 1})...")
            boosting_result = generate_boosting(train_df, test_df, rs)
            boosting_record = {
                "trial": trial,
                "best_params": boosting_result["best_params"],
                "cv_train_score": boosting_result["cv_train_score"],
                "cv_val_score": boosting_result["cv_val_score"],
                "test_accuracy": boosting_result["test_metrics"]["accuracy"],
                "test_metrics": boosting_result["test_metrics"],
                "y_test": boosting_result["y_test"].tolist(),
                "y_pred": boosting_result["y_pred"].tolist(),
                "y_proba": (
                    boosting_result["y_proba"].tolist()
                    if boosting_result["y_proba"] is not None
                    else None
                ),
            }
            results[BOOSTING_NAME][split_name].append(boosting_record)

            # Random Forest
            print(f"\n>>> Running Random Forest (Trial {trial + 1})...")
            rf_result = generate_random_forest(train_df, test_df, rs)
            rf_record = {
                "trial": trial,
                "best_params": rf_result["best_params"],
                "cv_train_score": rf_result["cv_train_score"],
                "cv_val_score": rf_result["cv_val_score"],
                "test_accuracy": rf_result["test_metrics"]["accuracy"],
                "test_metrics": rf_result["test_metrics"],
                "y_test": rf_result["y_test"].tolist(),
                "y_pred": rf_result["y_pred"].tolist(),
                "y_proba": (
                    rf_result["y_proba"].tolist()
                    if rf_result["y_proba"] is not None
                    else None
                ),
                "feature_importances": (
                    rf_result["feature_importances"].tolist()
                    if rf_result["feature_importances"] is not None
                    else None
                ),
                "feature_names": rf_result["feature_names"],
            }
            results[RANDOM_FOREST_NAME][split_name].append(rf_record)

            # Neural Network
            print(f"\n>>> Running Neural Network (Trial {trial + 1})...")
            nn_result = generate_neural_network(train_df, test_df, rs)
            nn_record = {
                "trial": trial,
                "best_params": nn_result["best_params"],
                "cv_train_score": nn_result["cv_train_score"],
                "cv_val_score": nn_result["cv_val_score"],
                "test_accuracy": nn_result["test_metrics"]["accuracy"],
                "test_metrics": nn_result["test_metrics"],
                "y_test": nn_result["y_test"].tolist(),
                "y_pred": nn_result["y_pred"].tolist(),
                "y_proba": (
                    nn_result["y_proba"].tolist()
                    if nn_result["y_proba"] is not None
                    else None
                ),
                "training_info": nn_result.get("training_info"),
            }
            results[NEURAL_NETWORK_NAME][split_name].append(nn_record)

    # 5. Save results
    results_dir = os.path.join(curr_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "wine_all_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[wine.py] Saved all results to {out_path}")

    # 6. Generate plots
    base_plots_dir = os.path.join(curr_dir, "../..", "plots/wine_plots/results")

    boosting_plots_dir = os.path.join(base_plots_dir, "boosting")
    plot_wine_boosting_summary(results[BOOSTING_NAME], boosting_plots_dir)

    rf_plots_dir = os.path.join(base_plots_dir, "random_forest")
    plot_wine_random_forest_summary(results[RANDOM_FOREST_NAME], rf_plots_dir)

    nn_plots_dir = os.path.join(base_plots_dir, "neural_network")
    plot_wine_neural_network_summary(results[NEURAL_NETWORK_NAME], nn_plots_dir)

    comparison_dir = os.path.join(base_plots_dir, "comparison")
    plot_wine_model_comparison(results, comparison_dir)


if __name__ == "__main__":
    main()
