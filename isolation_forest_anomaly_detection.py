"""
Isolation Forest with Grid Search and Multi-Seed Evaluation

This script performs anomaly detection using Isolation Forest on a synthetic non-Gaussian dataset.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
import pandas as pd

def generate_dataset():
    """
    Generate a synthetic non-Gaussian dataset for anomaly detection.
    Returns:
        X_train, X_test, y_train, y_test: Train-test splits
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        class_sep=2.0,
        random_state=42
    )
    y = np.where(y == 0, 1, -1)  # Map classes to 1 (normal) and -1 (anomalous)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

def grid_search_isolation_forest(X_train, X_test, y_train, y_test, random_seeds):
    """
    Perform grid search on Isolation Forest with multiple seeds for robust evaluation.
    Args:
        X_train, X_test, y_train, y_test: Train-test splits
        random_seeds: List of random seeds for multi-seed evaluation

    Returns:
        results_df: DataFrame containing grid search results
        best_params: Best parameters based on Mean AUC-ROC
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_samples": [0.5, 1.0],
        "contamination": [0.1, 0.2],  # Percentage of anomalies
        "max_features": [1.0, 0.8],
    }

    all_results = []
    for params in ParameterGrid(param_grid):
        seed_results = []
        for seed in random_seeds:
            model = IsolationForest(**params, random_state=seed)
            model.fit(X_train)

            y_scores = -model.decision_function(X_test)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 1, -1)  # Align labels to 1 (normal), -1 (anomalous)

            auc = roc_auc_score(y_test, y_scores)
            seed_results.append(auc)

        mean_auc = np.mean(seed_results)
        std_auc = np.std(seed_results)

        all_results.append({
            "Parameters": params,
            "Mean AUC-ROC": mean_auc,
            "Std AUC-ROC": std_auc,
        })

    results_df = pd.DataFrame(all_results)

    # Sort by Mean AUC-ROC
    results_df.sort_values(by="Mean AUC-ROC", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Identify best parameters
    best_row = results_df.iloc[0]
    best_params = best_row["Parameters"]

    return results_df, best_params

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    random_seeds = [42, 123, 456, 789, 101112]
    results_df, best_params = grid_search_isolation_forest(X_train, X_test, y_train, y_test, random_seeds)

    print(f"Best Parameters: {best_params}")
    best_row = results_df.iloc[0]
    print(f"Best Mean AUC-ROC: {best_row['Mean AUC-ROC']:.4f} ± {best_row['Std AUC-ROC']:.4f}")
    print("\nFull Results:")
    print(results_df)
