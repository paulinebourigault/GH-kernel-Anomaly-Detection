"""
Kernel-Based Anomaly Detection with GH Kernels and Standard Kernels

This script demonstrates the use of Generalized Hyperbolic (GH) kernels and standard kernels 
(RBF, Polynomial, Linear, Sigmoid) for anomaly detection using One-Class SVM.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.datasets import make_classification
import pandas as pd
from scipy.special import kv
import time

# Generate synthetic non-Gaussian dataset
def generate_data():
    """
    Generates a synthetic non-Gaussian dataset for anomaly detection.
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

# Vectorized GH Kernel function
def generalized_hyperbolic_kernel(X1, X2, params):
    """
    Compute the Generalized Hyperbolic (GH) kernel matrix.
    """
    lam, alpha, beta, delta, mu = (
        params["lambda"],
        params["alpha"],
        params["beta"],
        params["delta"],
        params["mu"],
    )
    gamma = np.sqrt(alpha**2 - beta**2)
    if alpha**2 <= beta**2:
        raise ValueError("Invalid GH parameters: alpha^2 must be > beta^2 for stability.")

    z = np.linalg.norm(X1[:, None] - X2, axis=2)
    bessel_term = kv(lam - 0.5, alpha * np.sqrt(delta**2 + z**2))
    bessel_term = np.nan_to_num(bessel_term, nan=1e-10, posinf=1e-10, neginf=1e-10)

    kernel_matrix = (
        (gamma / delta) ** lam
        / (np.sqrt(2 * np.pi) * kv(lam, delta * gamma))
        * np.exp(beta * z)
        * bessel_term
        * (np.sqrt(delta**2 + z**2) / alpha) ** (lam - 0.5)
    )
    return kernel_matrix

# Dedicated functions for standard kernels
def rbf_kernel(X1, X2, gamma):
    return np.exp(-gamma * np.linalg.norm(X1[:, None] - X2, axis=2)**2)

def polynomial_kernel(X1, X2, degree, gamma, coef0):
    return (gamma * X1.dot(X2.T) + coef0) ** degree

def linear_kernel(X1, X2):
    return X1.dot(X2.T)

def sigmoid_kernel(X1, X2, gamma, coef0):
    return np.tanh(gamma * X1.dot(X2.T) + coef0)

# GH Kernel configurations
gh_kernel_configs = {
    "Full GH Kernel": {"lambda": 1.0, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
    "GH Kernel (Gaussian)": {"lambda": 0.5, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
    "GH Kernel (NIG)": {"lambda": -0.5, "alpha": 2.0, "beta": 0.5, "delta": 1.0, "mu": 0.0},
    "GH Kernel (Student's t)": {"lambda": -1.0, "alpha": 2.0, "beta": 0.0, "delta": 1.0, "mu": 0.0},
    "GH Kernel (Hyperbolic)": {"lambda": 1.0, "alpha": 1.5, "beta": 0.3, "delta": 1.0, "mu": 0.0},
}

# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()

    # Combine kernel configurations
    kernel_methods = {
        "RBF": rbf_kernel,
        "Polynomial": polynomial_kernel,
        "Linear": linear_kernel,
        "Sigmoid": sigmoid_kernel,
    }

    kernel_hyperparams = {
        **{
            name: {
                "lambda": [params["lambda"] - 0.5, params["lambda"], params["lambda"] + 0.5],
                "alpha": [params["alpha"] * 0.8, params["alpha"], params["alpha"] * 1.2],
                "beta": [params["beta"] - 0.2, params["beta"], params["beta"] + 0.2],
                "delta": [params["delta"] * 0.8, params["delta"], params["delta"] * 1.2],
                "mu": [params["mu"] - 0.5, params["mu"], params["mu"] + 0.5],
            }
            for name, params in gh_kernel_configs.items()
        },
        **{
            "RBF": {"gamma": [0.1, 0.5, 1.0]},
            "Polynomial": {"degree": [2, 3, 4], "gamma": [0.1, 0.5], "coef0": [0, 1]},
            "Linear": {},  # No hyperparameters
            "Sigmoid": {"gamma": [0.1, 0.5], "coef0": [0, 1]},
        },
    }

    results = []

    # Evaluate kernels
    for kernel_name, kernel_func in {**kernel_methods, **gh_kernel_configs}.items():
        param_grid = ParameterGrid(kernel_hyperparams.get(kernel_name, [{}]))
        best_score = 0
        best_params = None
        start_time = time.time()

        for params in param_grid:
            try:
                if kernel_name in gh_kernel_configs:
                    K_train = generalized_hyperbolic_kernel(X_train, X_train, params)
                    K_test = generalized_hyperbolic_kernel(X_test, X_train, params)
                else:
                    K_train = kernel_func(X_train, X_train, **params)
                    K_test = kernel_func(X_test, X_train, **params)

                ocs = OneClassSVM(kernel="precomputed", nu=0.1)
                ocs.fit(K_train)

                y_scores = ocs.decision_function(K_test)
                auc = roc_auc_score(y_test, y_scores)

                if auc > best_score:
                    best_score = auc
                    best_params = params

            except Exception as e:
                print(f"Error with kernel {kernel_name} and params {params}: {e}")

        training_time = time.time() - start_time
        results.append({"Kernel": kernel_name, "Best Parameters": best_params, "Best AUC-ROC": best_score, "Training Time (s)": training_time})

    results_df = pd.DataFrame(results)
    print(results_df)
