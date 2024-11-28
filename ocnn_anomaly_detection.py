"""
OC-NN with Grid Search for Anomaly Detection

This script implements an OC-NN (One-Class Neural Network) for anomaly detection on a synthetic non-Gaussian dataset.
It performs a grid search over hyperparameters to find the best configuration based on AUC-ROC.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import pandas as pd

def generate_dataset():
    """
    Generate a synthetic non-Gaussian dataset for anomaly detection.
    Returns:
        Train-test splits: X_train, X_test, y_train, y_test
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
    y = np.where(y == 0, 1, -1)  # Map classes: 1 (normal), -1 (anomalous)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

class OCNN(nn.Module):
    """
    One-Class Neural Network (OC-NN) model for anomaly detection.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(OCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def ocnn_loss(output, nu):
    """
    Compute OC-NN centered hypersphere loss.
    Args:
        output: Model output
        nu: Hyperparameter for anomaly detection
    Returns:
        Loss value
    """
    term1 = 0.5 * torch.sum(output**2)
    term2 = 1 / nu * torch.mean(torch.relu(1 - output))
    term3 = -torch.mean(output)
    return term1 + term2 + term3

def train_ocnn(model, train_loader, optimizer, nu, n_epochs):
    """
    Train OC-NN model.
    Args:
        model: OC-NN model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        nu: Hyperparameter for anomaly detection
        n_epochs: Number of training epochs
    """
    model.train()
    for epoch in range(n_epochs):
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = ocnn_loss(output, nu)
            loss.backward()
            optimizer.step()

def evaluate_ocnn(model, test_loader):
    """
    Evaluate OC-NN model.
    Args:
        model: Trained OC-NN model
        test_loader: DataLoader for test data
    Returns:
        Array of anomaly scores
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            output = model(x)
            scores.extend(output.squeeze().numpy())
    return np.array(scores)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_dataset()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    param_grid = {
        "hidden_dim": [8, 16, 32],
        "lr": [0.001, 0.01],
        "nu": [0.1, 0.2],
        "n_epochs": [20, 30, 50]
    }

    best_auc = 0
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        model = OCNN(input_dim=X_train.shape[1], hidden_dim=params["hidden_dim"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        train_ocnn(model, train_loader, optimizer, nu=params["nu"], n_epochs=params["n_epochs"])
        
        scores = evaluate_ocnn(model, test_loader)
        auc = roc_auc_score(y_test, -scores)  # Negative scores for anomaly detection
        
        results.append({"Parameters": params, "AUC-ROC": auc})
        
        if auc > best_auc:
            best_auc = auc
            best_params = params

    results_df = pd.DataFrame(results)
    print(f"Best Parameters: {best_params}")
    print(f"Best AUC-ROC: {best_auc:.4f}")
    print("\nFull Results:")
    print(results_df.sort_values(by="AUC-ROC", ascending=False))