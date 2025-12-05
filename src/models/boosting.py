# src/models/boosting.py

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _build_preprocessing_and_model(num_cols, cat_cols, problem_type, random_state):
    """
    Create a preprocessing + model pipeline.

    - num_cols -> StandardScaler
    - cat_cols -> OneHotEncoder
    - model   -> GradientBoostingClassifier or Regressor
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if problem_type == "classification":
        model = GradientBoostingClassifier(random_state=random_state)
    elif problem_type == "regression":
        model = GradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def run_boosting_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list,
    target_col: str,
    problem_type: str = "classification",   # or "regression"
    random_state: int = 42,
    param_grid: dict | None = None,
):
    """
    Generic boosting experiment:
    - Uses ONLY train_df for GridSearchCV (10-fold)
    - Refits best model on full train_df
    - Evaluates once on held-out test_df
    - Returns dict with model, best params, CV summary, and test metrics
    """

    np.random.seed(random_state)

    # =========================
    # 1. Split X / y
    # =========================
    X_train = train_df[predictors].copy()
    X_test = test_df[predictors].copy()

    y_train_raw = train_df[target_col].copy()
    y_test_raw = test_df[target_col].copy()

    # For classification, encode labels to 0/1 (or 0..K-1)
    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
    else:
        y_train = y_train_raw.values
        y_test = y_test_raw.values

    # Identify numeric & categorical columns
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(X_train[c])]

    # =========================
    # 2. Build base pipeline
    # =========================
    pipeline = _build_preprocessing_and_model(
        num_cols=num_cols,
        cat_cols=cat_cols,
        problem_type=problem_type,
        random_state=random_state,
    )

    # =========================
    # 3. Define hyperparameter grid
    # =========================
    if param_grid is None:
        # Reasonable defaults; you can override from experiments/bank.py
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [1, 3],
        }

    # Choose scoring
    if problem_type == "classification":
        scoring = "roc_auc"
    else:
        scoring = "neg_mean_squared_error"

    # =========================
    # 4. Grid search (10-fold CV) on TRAIN ONLY
    # =========================
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,       # refit best model on full train set
        verbose=1,
        return_train_score=True,
    )

    print("\n=== GRID SEARCH: Gradient Boosting ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Hyperparameter combinations: "
          f"{len(param_grid['model__n_estimators']) * len(param_grid['model__learning_rate']) * len(param_grid['model__max_depth'])}")
    print("Fitting GridSearchCV (10-fold CV on TRAIN ONLY)...")

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_          # mean val score across folds
    best_index = grid_search.best_index_
    best_cv_std = grid_search.cv_results_["std_test_score"][best_index]

     # Mean train/val scores for the best hyperparameter combo
    cv_train_score = grid_search.cv_results_["mean_train_score"][best_index]
    cv_val_score = grid_search.cv_results_["mean_test_score"][best_index]

    print("\n=== GRID SEARCH COMPLETE ===")
    print("Best params:", best_params)
    print(f"Best CV score ({scoring}): {best_cv_score:.4f} ± {best_cv_std:.4f}")

    # =========================
    # 5. Train metrics (overfitting check)
    # =========================
    # best_estimator is already refit on ALL train data
    y_train_pred = best_estimator.predict(X_train)

    train_metrics = {}
    if problem_type == "classification":
        train_metrics["accuracy"] = accuracy_score(y_train, y_train_pred)
        train_metrics["precision"] = precision_score(y_train, y_train_pred, zero_division=0)
        train_metrics["recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        train_metrics["f1"] = f1_score(y_train, y_train_pred, zero_division=0)
    else:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_metrics.update({
            "mse": train_mse,
            "rmse": train_rmse,
            "mae": train_mae,
            "r2": train_r2,
        })

    # =========================
    # 6. Evaluate on held-out TEST set
    # =========================
    print("\n=== EVALUATING ON TEST SET (HELD-OUT) ===")
    y_pred = best_estimator.predict(X_test)

    test_metrics = {}
    y_proba = None   # <= will store probabilities for ROC curve

    if problem_type == "classification":
        # If classifier supports predict_proba, compute AUC + keep scores for ROC curve
        if hasattr(best_estimator.named_steps["model"], "predict_proba"):
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = np.nan
        else:
            roc_auc = np.nan

        test_metrics["accuracy"] = accuracy_score(y_test, y_pred)
        test_metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        test_metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        test_metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)
        test_metrics["roc_auc"] = roc_auc

        print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall   : {test_metrics['recall']:.4f}")
        print(f"Test F1       : {test_metrics['f1']:.4f}")
        print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        test_metrics["mse"] = mse
        test_metrics["rmse"] = rmse
        test_metrics["mae"] = mae
        test_metrics["r2"] = r2

        print(f"Test R²  : {r2:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE : {mae:.4f}")

    # =========================
    # 7. Pack results in a dict
    # =========================
    results = {
        "model": best_estimator,
        "best_params": best_params,

        # CV info
        "cv_primary_metric": best_cv_score,   # same as best_cv_score
        "cv_scoring": scoring,
        "cv_train_score": float(cv_train_score),  # mean train score for best params
        "cv_val_score": float(cv_val_score),      # mean val score for best params
        "grid_search": grid_search,

        # Test metrics & outputs
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test_raw": y_test_raw,

        # Meta
        "predictors": predictors,
        "target_col": target_col,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
    }

    return results