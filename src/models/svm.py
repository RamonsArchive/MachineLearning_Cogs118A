# src/models/svm.py
"""
Support Vector Machine (SVM) Classifier for Bank Marketing Dataset
- Uses SVC for binary classification
- StandardScaler for numeric features, OneHotEncoder for categorical
- GridSearchCV with 10-fold CV on training data only
- No data leakage: test set never touched during hyperparameter tuning

SVM HYPERPARAMETERS EXPLAINED:
------------------------------
1. kernel: The kernel function transforms data into higher dimensions
   - 'linear': K(x,y) = x·y - Best for linearly separable data, fastest
   - 'rbf': K(x,y) = exp(-gamma||x-y||²) - Most flexible, handles non-linear patterns
   - 'poly': K(x,y) = (gamma*x·y + coef0)^degree - Polynomial transformation
   - 'sigmoid': K(x,y) = tanh(gamma*x·y + coef0) - Similar to neural network
   
   For high-dimensional data (after one-hot encoding), 'linear' often works well.
   For non-linear patterns, 'rbf' is the go-to choice.

2. C: Regularization parameter (inverse of regularization strength)
   - Controls trade-off between smooth decision boundary and correct classification
   - Low C (0.01-1): More regularization, smoother boundary, may underfit
   - High C (10-1000): Less regularization, tighter fit, may overfit
   - Typical range: 0.1 to 100

3. gamma: Kernel coefficient (for 'rbf', 'poly', 'sigmoid')
   - Defines influence radius of training examples
   - High gamma: Only nearby points influence decision, complex boundaries
   - Low gamma: Far points also influence, smoother boundaries
   - 'scale': 1 / (n_features * X.var()) - Recommended default
   - 'auto': 1 / n_features
   - Numeric: 0.001 to 10

4. degree: Polynomial degree (only for 'poly' kernel)
   - Higher degree = more complex decision boundary
   - Typical: 2-5

RECOMMENDATIONS FOR BANK DATASET:
- Start with RBF kernel (most flexible)
- Also test linear kernel (faster, works well for high-dimensional data)
- Focus on tuning C and gamma
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.svm import SVC, SVR
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


def _build_preprocessing_and_model(num_cols, cat_cols, problem_type, random_state,
                                    kernel='rbf', probability=True):
    """
    Create a preprocessing + model pipeline.

    - num_cols -> StandardScaler (crucial for SVM!)
    - cat_cols -> OneHotEncoder
    - model   -> SVC or SVR
    
    Note: StandardScaler is CRITICAL for SVM because it uses distance-based
    calculations. Features on different scales will dominate the kernel.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if problem_type == "classification":
        model = SVC(
            kernel=kernel,
            probability=False,  # done enable predict_proba. Do ROC curves with decision func
            random_state=random_state,
            cache_size=1000,  # MB, increase for faster training
        )
    elif problem_type == "regression":
        model = SVR(
            kernel=kernel,
            cache_size=1000,
        )
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def run_svm_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list,
    target_col: str,
    problem_type: str = "classification",
    random_state: int = 42,
    param_grid: dict | None = None,
):
    """
    SVM experiment:
    - Uses ONLY train_df for GridSearchCV (10-fold)
    - Refits best model on full train_df
    - Evaluates once on held-out test_df
    - Returns dict with model, best params, CV summary, and test metrics
    
    NO DATA LEAKAGE: Test set is never used during hyperparameter tuning.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (held-out, only for final evaluation)
        predictors: List of predictor column names
        target_col: Target column name
        problem_type: "classification" or "regression"
        random_state: Random seed
        param_grid: Dict of hyperparameters to tune. Recommended:
            {
                "model__kernel": ['rbf', 'linear'],
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ['scale', 0.01, 0.1, 1],  # ignored for linear
            }
    """

    np.random.seed(random_state)

    # =========================
    # 1. Split X / y
    # =========================
    X_train = train_df[predictors].copy()
    X_test = test_df[predictors].copy()

    y_train_raw = train_df[target_col].copy()
    y_test_raw = test_df[target_col].copy()

    # For classification, encode labels to 0/1
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
        # Default grid balancing thoroughness and speed
        # RBF kernel with gamma tuning + linear kernel
        param_grid = {
            "model__kernel": ['rbf', 'linear'],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ['scale', 0.01, 0.1],  # ignored when kernel='linear'
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

    # Calculate total combinations for logging
    total_combos = 1
    for key, values in param_grid.items():
        total_combos *= len(values)

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

    print("\n=== GRID SEARCH: Support Vector Machine ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Hyperparameter combinations: {total_combos}")
    print(f"Kernels being tested: {param_grid.get('model__kernel', ['rbf'])}")
    print("Fitting GridSearchCV (10-fold CV on TRAIN ONLY)...")
    print("Note: SVM can be slow on large datasets. Using cache_size=1000MB.")

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
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
    y_proba = None

    if problem_type == "classification":
        # Use decision_function for ROC-AUC (faster than predict_proba!)
        # decision_function returns signed distance to hyperplane - works perfectly for ranking
        if hasattr(best_estimator.named_steps["model"], "decision_function"):
            y_proba = best_estimator.decision_function(X_test)
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = np.nan
        elif hasattr(best_estimator.named_steps["model"], "predict_proba"):
            # Fallback to predict_proba if decision_function not available
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = np.nan
        else:
            y_proba = None
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
    # 7. Get SVM-specific info
    # =========================
    final_model = best_estimator.named_steps["model"]
    svm_info = {
        "kernel": best_params.get("model__kernel", final_model.kernel),
        "C": best_params.get("model__C", final_model.C),
        "gamma": best_params.get("model__gamma", final_model.gamma),
        "n_support": final_model.n_support_.tolist() if hasattr(final_model, 'n_support_') else None,
        "total_support_vectors": int(np.sum(final_model.n_support_)) if hasattr(final_model, 'n_support_') else None,
    }

    # =========================
    # 8. Pack results in a dict
    # =========================
    results = {
        "model": best_estimator,
        "best_params": best_params,

        # CV info
        "cv_primary_metric": best_cv_score,
        "cv_scoring": scoring,
        "cv_train_score": float(cv_train_score),
        "cv_val_score": float(cv_val_score),
        "grid_search": grid_search,

        # Test metrics & outputs
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_test_raw": y_test_raw,

        # SVM-specific info
        "svm_info": svm_info,

        # Meta
        "predictors": predictors,
        "target_col": target_col,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
    }

    return results

