# Empirical Comparison of Classification and Regression Algorithms

**Authors:**
- Ramon Calderon McDargh-Mitchell
  - Email: rmcdeem.m@gmail.com
  - Email: clutchdev.apps@gmail.com

## Abstract

This repository contains the code and results for an empirical comparison of machine learning algorithms—gradient-boosted trees (XGBoost), Random Forests, and Neural Networks—following the methodology of Caruana and Niculescu-Mizil. Three classification models were evaluated on three datasets, and three regression models were evaluated on two datasets, using 20/80, 50/50, and 80/20 train–test splits averaged over three independent trials.

For classification, XGBoost achieved the strongest performance on two of the three datasets, while Random Forest performed best on the remaining dataset. Performance was evaluated using F1 score as the primary metric with ROC–AUC as the secondary metric. For regression tasks, XGBoost achieved the lowest error on one dataset, whereas Random Forest performed best on the other. Regression performance was primarily evaluated using test RMSE, with R² as the secondary metric.

## File Structure

```
MachineLearning_Cogs118A/
├── src/
│   ├── main.py                    # Main entry point to run all experiments
│   ├── experiments/               # Individual experiment scripts
│   │   ├── bank.py               # Bank Marketing classification experiment
│   │   ├── thyroid_cancer.py     # Thyroid Cancer Recurrence classification experiment
│   │   ├── wine.py               # Wine Quality classification experiment
│   │   ├── face_temp.py          # Infrared Thermography regression experiment
│   │   └── parkinsons.py         # Parkinson's Telemonitoring regression experiment
│   ├── models/                    # Model implementations
│   │   ├── boosting.py           # XGBoost (gradient-boosted trees)
│   │   ├── random_forest.py     # Random Forest
│   │   ├── neural_net.py         # Neural Networks (MLP)
│   │   ├── svm.py                # Support Vector Machines (not included in final results)
│   │   └── elastic_net.py       # ElasticNet (linear baseline for regression)
│   ├── utils/                     # Utility functions
│   │   ├── load/                 # Data loading functions
│   │   ├── clean/                # Data cleaning functions
│   │   └── eda/                  # Exploratory Data Analysis functions
│   └── graphs/                    # Plotting and visualization functions
│       ├── bank_*_plots.py       # Bank experiment plots
│       ├── thyroid_*_plots.py    # Thyroid experiment plots
│       ├── wine_*_plots.py       # Wine experiment plots
│       ├── face_temp_plots.py    # Face temperature regression plots
│       └── parkinsons_plots.py   # Parkinson's regression plots
├── plots/                         # Generated plots and reports
│   ├── bank_plots/
│   ├── thyroid_plots/
│   ├── wine_plots/
│   ├── face_temp_plots/
│   └── parkinsons_plots/
├── datasets/                      # Dataset files (not included in repository)
└── results/                       # JSON results files
```

## Experimental Pipeline

### Methodology

1. **Data Loading**: Each dataset is loaded from the `datasets/` directory
2. **Data Cleaning**: Missing values, data leakage, and preprocessing are handled
3. **Exploratory Data Analysis (EDA)**: Naive baselines, class distributions, and data characteristics are analyzed
4. **Train-Test Splits**: Three different splits are evaluated:
   - 20/80: 20% training, 80% testing
   - 50/50: 50% training, 50% testing
   - 80/20: 80% training, 20% testing
5. **Trials**: Each split is repeated 3 times with different random seeds to reduce variance
6. **Hyperparameter Tuning**: 5-fold cross-validation (stratified for classification) is used for grid search
7. **Model Evaluation**: Best model is refit on full training set and evaluated on held-out test set
8. **Results**: Metrics, plots, and reports are generated and saved

### Running Experiments

#### Run All Experiments

To run all experiments, simply execute:

```bash
python src/main.py
```

This will run all active experiments (bank, thyroid_cancer, wine) sequentially.

#### Run Individual Experiments

To run a specific experiment:

```bash
# Bank Marketing classification
python src/experiments/bank.py

# Thyroid Cancer Recurrence classification
python src/experiments/thyroid_cancer.py

# Wine Quality classification
python src/experiments/wine.py

# Face Temperature regression
python src/experiments/face_temp.py

# Parkinson's Telemonitoring regression
python src/experiments/parkinsons.py
```

### Model Families

1. **XGBoost (Gradient-Boosted Trees)**: Modern gradient boosting with regularization
2. **Random Forest**: Ensemble of decision trees with bagging
3. **Neural Networks (MLP)**: Multi-layer perceptrons with various architectures

### Evaluation Metrics

**Classification:**
- Primary: F1 Score (weighted for multiclass)
- Secondary: ROC-AUC (when applicable)

**Regression:**
- Primary: RMSE (Root Mean Squared Error)
- Secondary: R² (Coefficient of Determination), MAE (Mean Absolute Error)

## Results Summary

### Classification Results

**Table 1: Classification Performance (Test F1 / ROC–AUC, 80/20 split averaged across 3 trials)**

| Algorithm | Bank Marketing | Thyroid Cancer Recurrence | Wine |
|-----------|---------------|---------------------------|------|
| XGBoost | 0.516 / 0.803 | 0.929 / 0.992 | 1.000 / 1.000 |
| Random Forest | 0.507 / 0.797 | 0.935 / 0.987 | 0.991 / 1.000 |
| Neural Network | 0.351 / 0.792 | 0.901 / 0.979 | 0.972 / 1.000 |

**Key Findings:**
- XGBoost achieved the strongest performance on Bank Marketing and Wine datasets
- Random Forest performed best on Thyroid Cancer Recurrence dataset
- Neural networks consistently underperformed relative to tree-based methods

### Regression Results

**Table 2: Regression Performance (Test RMSE / R², 80/20 split averaged across 3 trials)**

| Algorithm | Infrared Thermography | Parkinson's Telemonitoring |
|-----------|----------------------|---------------------------|
| XGBoost | 0.232 / 0.736 | 1.430 / 0.981 |
| Random Forest | 0.231 / 0.735 | 5.029 / 0.768 |
| Neural Network | 0.300 / 0.546 | 3.780 / 0.869 |

**Key Findings:**
- Random Forest achieved the lowest RMSE on Infrared Thermography dataset
- XGBoost achieved the lowest RMSE on Parkinson's Telemonitoring dataset
- Neural networks showed consistent overfitting across regression tasks

## Output Files

After running experiments, results are saved in:

- **JSON Results**: `src/results/{dataset}_all_results.json`
- **Plots**: `plots/{dataset}_plots/results/{model}/`
  - Accuracy/R² plots
  - ROC curves (classification)
  - Confusion matrices (classification)
  - Residual plots (regression)
  - Feature importance plots
- **Text Reports**: `plots/{dataset}_plots/results/{model}/{dataset}_{model}_report.txt`
- **Comparison Reports**: `plots/{dataset}_plots/results/comparison/`

## Requirements

See `requirements.txt` for Python package dependencies.

## References

[1] R. Caruana and A. Niculescu-Mizil. An empirical comparison of supervised learning algorithms. In Proceedings of the 23rd International Conference on Machine Learning (ICML), pages 161–168, 2006.

[2] D. Dua and C. Graff. UCI Machine Learning Repository. http://archive.ics.uci.edu/ml, 2019.

## License

See LICENSE file for details.

