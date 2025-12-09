# src/models/elastic_net.py
"""
ElasticNet Regression - Unified Linear Baseline

ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization:
    Loss = MSE + α * (ρ * Σ|w| + (1-ρ)/2 * Σw²)

Where:
    - α (alpha): Overall regularization strength
    - ρ (l1_ratio): Mix between L1 and L2
        - l1_ratio=0 → Pure Ridge (L2)
        - l1_ratio=1 → Pure Lasso (L1)
        - l1_ratio=0.5 → 50/50 mix

By tuning l1_ratio, we automatically find the best linear baseline!
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _build_preprocessing_and_model(num_cols, cat_cols, random_state):
    """Create preprocessing + ElasticNet pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = ElasticNet(
        random_state=random_state,
        max_iter=10000,  # Ensure convergence
        warm_start=False,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def run_elastic_net_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list,
    target_col: str,
    random_state: int = 42,
    param_grid: dict | None = None,
):
    """
    Run ElasticNet regression as the LINEAR BASELINE.
    
    Automatically finds best mix of L1/L2 regularization.
    If ElasticNet performs close to RF/XGBoost, relationship is mostly linear.
    """
    np.random.seed(random_state)

    # =========================
    # 1. Split X / y
    # =========================
    X_train = train_df[predictors].copy()
    X_test = test_df[predictors].copy()

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Identify numeric & categorical columns
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(X_train[c])]

    # =========================
    # 2. Build base pipeline
    # =========================
    pipeline = _build_preprocessing_and_model(
        num_cols=num_cols,
        cat_cols=cat_cols,
        random_state=random_state,
    )

    # =========================
    # 3. Define hyperparameter grid
    # =========================
    if param_grid is None:
        param_grid = {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],  # 0=Ridge, 1=Lasso
        }

    scoring = "neg_mean_squared_error"

    # =========================
    # 4. Grid search (5-fold CV)
    # =========================
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )

    print(f"\n=== GRID SEARCH: ELASTICNET (Linear Baseline) ===")
    print(f"Training samples: {len(X_train)}")
    print("l1_ratio: 0=Ridge(L2), 1=Lasso(L1), 0.5=Mix")
    print("Fitting GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    best_index = grid_search.best_index_

    cv_train_score = grid_search.cv_results_["mean_train_score"][best_index]
    cv_val_score = grid_search.cv_results_["mean_test_score"][best_index]

    # Interpret the best l1_ratio
    l1_ratio = best_params.get("model__l1_ratio", 0.5)
    if l1_ratio == 0:
        reg_type = "Pure Ridge (L2)"
    elif l1_ratio == 1:
        reg_type = "Pure Lasso (L1)"
    else:
        reg_type = f"Mix ({l1_ratio:.0%} L1, {1-l1_ratio:.0%} L2)"

    print("\n=== GRID SEARCH COMPLETE ===")
    print("Best params:", best_params)
    print(f"Best regularization: {reg_type}")
    print(f"Best CV score ({scoring}): {best_cv_score:.4f}")

    # =========================
    # 5. Evaluate on TEST set
    # =========================
    print("\n=== EVALUATING ON TEST SET ===")
    y_pred = best_estimator.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    test_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    print(f"Test R²  : {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

    # =========================
    # 6. Get coefficients
    # =========================
    model = best_estimator.named_steps["model"]
    coefficients = model.coef_
    
    try:
        preprocessor = best_estimator.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = None

    # Count non-zero coefficients (Lasso sparsity)
    n_nonzero = np.sum(coefficients != 0)
    print(f"Non-zero coefficients: {n_nonzero}/{len(coefficients)} ({100*n_nonzero/len(coefficients):.1f}%)")

    # =========================
    # 7. Pack results
    # =========================
    results = {
        "model": best_estimator,
        "best_params": best_params,
        "cv_scoring": scoring,
        "cv_train_score": float(cv_train_score),
        "cv_val_score": float(cv_val_score),
        "grid_search": grid_search,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "coefficients": coefficients,
        "feature_names": feature_names.tolist() if feature_names is not None else None,
        "predictors": predictors,
        "target_col": target_col,
        "regularization_type": reg_type,
        "n_nonzero_coefs": int(n_nonzero),
    }

    return results

