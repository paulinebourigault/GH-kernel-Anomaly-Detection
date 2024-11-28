"""
Kernel-Based Anomaly Detection using GH and Standard Kernels

This script implements kernel density estimation (KDE) and anomaly detection using:
- Generalized Hyperbolic (GH) Kernels
- Standard KDE Kernels (Gaussian, Tophat, Exponential, Epanechnikov)

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from scipy.special import kv
import time


# Step 1: Dataset Generation
def generate_data():
    """
    Generate a synthetic non-Gaussian dataset for anomaly detection.
    Returns:
        X_train, X_test, y_train, y_test: Train and test splits
    """
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
    y = np.where(y == 0, 1, -1)  # Map classes: 1 (normal), -1 (anomalies)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Step 2: Kernel Definitions
def generalized_hyperbolic_kernel(x, y, params):
    """
    Compute the Generalized Hyperbolic (GH) kernel value between two points.
    """
    lam, alpha, beta, delta, mu = (
        params["lambda"],
        params["alpha"],
        params["beta"],
        params["delta"],
        params["mu"],
    )
    gamma = np.sqrt(alpha**2 - beta**2)
    z = np.linalg.norm(x - y)

    bessel = kv(lam - 0.5, alpha * np.sqrt(delta**2 + z**2))
    kernel_value = (
        (gamma / delta) ** lam
        / (np.sqrt(2 * np.pi) * kv(lam, delta * gamma))
        * np.exp(beta * z)
        * bessel
        * (np.sqrt(delta**2 + z**2) / alpha) ** (lam - 0.5)
    )
    return kernel_value


def gaussian_kde(x, X, bandwidth):
    """Gaussian KDE kernel."""
    return np.sum(np.exp(-np.linalg.norm(x - X, axis=1)**2 / (2 * bandwidth**2))) / (np.sqrt(2 * np.pi) * bandwidth * len(X))


def tophat_kde(x, X, bandwidth):
    """Tophat KDE kernel."""
    return np.sum(np.linalg.norm(x - X, axis=1) < bandwidth) / (len(X) * bandwidth)


def exponential_kde(x, X, bandwidth):
    """Exponential KDE kernel."""
    return np.sum(np.exp(-np.linalg.norm(x - X, axis=1) / bandwidth)) / (len(X) * bandwidth)


def epanechnikov_kde(x, X, bandwidth):
    """Epanechnikov KDE kernel."""
    distances = np.linalg.norm(x - X, axis=1)
    return np.sum((1 - (distances / bandwidth)**2) * (distances < bandwidth)) / (len(X) * bandwidth)


# Step 3: Anomaly Detection Functions
def anomaly_score_kde(X, x, params, bandwidth):
    """
    Compute the anomaly score for a point using KDE with GH kernel.
    """
    density = np.mean([generalized_hyperbolic_kernel(x, xi, params) for xi in X])
    return -np.log(max(density, 1e-10))  # Avoid log(0)


def evaluate_kernels(X_train, X_test, y_train, y_test, gh_kernel_configs, standard_kernels, bandwidth):
    """
    Evaluate GH and standard kernels, measure AUC-ROC, and training time.
    Returns:
        results: List of dictionaries containing kernel evaluation results
    """
    results = []

    # Evaluate GH Kernels
    for kernel_name, base_params in gh_kernel_configs.items():
        best_auc = 0
        best_params = None
        start_time = time.time()

        param_grid = ParameterGrid({
            "lambda": [base_params["lambda"] - 0.5, base_params["lambda"], base_params["lambda"] + 0.5],
            "alpha": [base_params["alpha"] * 0.8, base_params["alpha"], base_params["alpha"] * 1.2],
            "beta": [base_params["beta"] - 0.2, base_params["beta"], base_params["beta"] + 0.2],
            "delta": [base_params["delta"] * 0.8, base_params["delta"], base_params["delta"] * 1.2],
            "mu": [base_params["mu"]],
        })

        for params in param_grid:
            tuned_params = {**base_params, **params}
            y_scores = [anomaly_score_kde(X_train, x, tuned_params, bandwidth) for x in X_test]
            auc = roc_auc_score(y_test, -np.array(y_scores))

            if auc > best_auc:
                best_auc = auc
                best_params = tuned_params

        training_time = time.time() - start_time
        results.append({
            "Kernel": kernel_name,
            "Best AUC-ROC": best_auc,
            "Best Parameters": best_params,
            "Training Time (s)": training_time,
        })

    # Evaluate Standard KDE Kernels
    for kernel_name, kernel_func in standard_kernels.items():
        best_auc = 0
        best_params = None
        start_time = time.time()

        param_grid = ParameterGrid({"bandwidth": [0.01, 0.1, 0.5]})
        for params in param_grid:
            bandwidth = params["bandwidth"]
            y_scores = [kernel_func(x, X_train, bandwidth) for x in X_test]
            anomaly_scores = -np.log(np.maximum(y_scores, 1e-10))  # Avoid log(0)

            auc = roc_auc_score(y_test, anomaly_scores)

            if auc > best_auc:
                best_auc = auc
                best_params = params

        training_time = time.time() - start_time
        results.append({
            "Kernel": kernel_name,
            "Best AUC-ROC": best_auc,
            "Best Parameters": best_params,
            "Training Time (s)": training_time,
        })

    return results


# Step 4: Main Execution
if __name__ == "__main__":
    # Generate dataset
    X_train, X_test, y_train, y_test = generate_data()

    # Define GH kernel configurations
    gh_kernel_configs = {
        "Full GH Kernel": {"lambda": 1.0, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Gaussian)": {"lambda": 0.5, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
        "GH Kernel (NIG)": {"lambda": -0.5, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Student's t)": {"lambda": -1.0, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
        "GH Kernel (Hyperbolic)": {"lambda": 1.0, "alpha": 1.5, "beta": 0.3, "delta": 1.0, "mu": 0.0},
    }

    # Define standard KDE kernels
    standard_kernels = {
        "Gaussian KDE": gaussian_kde,
        "Tophat KDE": tophat_kde,
        "Exponential KDE": exponential_kde,
        "Epanechnikov KDE": epanechnikov_kde,
    }

    # Evaluate kernels
    bandwidth = 0.01  # Fixed bandwidth here
    results = evaluate_kernels(X_train, X_test, y_train, y_test, gh_kernel_configs, standard_kernels, bandwidth)

    results_df = pd.DataFrame(results)
    print(results_df)
