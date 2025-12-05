import os
import sys
import json
# Add the src directory to the Python path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.load.load_bank import load_bank_data
from utils.clean.clean_bank import clean_bank
from utils.eda.eda_bank import eda_bank
from sklearn.model_selection import train_test_split

from models.boosting import run_boosting_experiment

from graphs.bank_boosting_plots import plot_bank_boosting_summary



RANDOM_STATE = 42
BOOSTING_NAME = "boosting"
RANDOM_FOREST_NAME = "random_forest"
NEURAL_NETWORK_NAME = "neural_network"
SVM_NAME = "svm"

def generate_boosting(train_df, test_df, RANDOM_STATE):
    target_col = "y"
    predictors = [c for c in train_df.columns if c != target_col]
    param_grid = {
        "model__n_estimators": [100, 200, 300, 600, 1000, 1200],
        "model__learning_rate": [0.001, 0.005, 0.01, 0.03, 0.1],
        "model__max_depth": [1, 2, 3, 5, 7, 9],
    }
    # param_grid = {
    #     "model__n_estimators": [100],
    #     "model__learning_rate": [0.001],
    #     "model__max_depth": [1],
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
    return train_df, test_df

def generate_neural_network(train_df, test_df, RANDOM_STATE):
    return train_df, test_df

def generate_svm(train_df, test_df, RANDOM_STATE):
    return train_df, test_df



def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_bank_data(curr_dir)
    clean_df = clean_bank(data)
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

            # ----- Run Boosting -----
            boosting_result = generate_boosting(train_df, test_df, rs)

            # EXPECTED fields in boosting_result (you can tweak names to your impl):
            # {
            #   "best_params": {...},
            #   "cv_best_score": float,
            #   "train_accuracy": float,
            #   "val_accuracy": float,
            #   "test_accuracy": float,
            #   ... (other metrics ok)
            # }

            trial_record = {
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

            results[BOOSTING_NAME][split_name].append(trial_record)

    # Save results to JSON
    out_path = os.path.join(curr_dir, "..", "results", "bank_boosting_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot results
    plots_dir = os.path.join(curr_dir, "../..", "plots/bank_plots", "bank_boosting_plots")
    plot_bank_boosting_summary(results[BOOSTING_NAME], plots_dir)


if __name__ == "__main__":
    main()