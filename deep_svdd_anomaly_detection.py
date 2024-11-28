"""
Deep SVDD with Grid Search for Anomaly Detection

This script implements the Deep Support Vector Data Description (Deep SVDD) framework for anomaly detection
on a synthetic dataset.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Step 1: Dataset Generation
def generate_data():
    """
    Generate a synthetic non-Gaussian dataset for anomaly detection.
    Returns:
        X_train, X_test, y_train, y_test: Train and test splits
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

# Step 2: Deep SVDD Model Definition
class DeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description (Deep SVDD) Model
    """
    def __init__(self, input_dim, hidden_dim):
        super(DeepSVDD, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)

# Step 3: Center Initialization
def initialize_center(deep_svdd, train_loader):
    """
    Initialize the hypersphere center as the mean of the network's outputs on the training data.
    """
    for layer in reversed(deep_svdd.network):
        if isinstance(layer, nn.Linear):
            c = torch.zeros(layer.out_features)
            break
    else:
        raise ValueError("No linear layer found in the Deep SVDD network.")

    n_samples = 0
    deep_svdd.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, _ = batch
            z = deep_svdd(x)
            n_samples += z.size(0)
            c += z.sum(dim=0)
    c /= n_samples
    return c

# Step 4: Training Function
def train_deep_svdd(model, center, train_loader, n_epochs=50, lr=1e-3):
    """
    Train the Deep SVDD model using the specified number of epochs and learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            x, _ = batch
            optimizer.zero_grad()
            z = model(x)
            # Deep SVDD objective: minimize distance to center
            loss = torch.mean(torch.sum((z - center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)

# Step 5: Evaluation Function
def evaluate_deep_svdd(model, center, test_loader):
    """
    Evaluate the Deep SVDD model using AUC-ROC.
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            z = model(x)
            dist = torch.sum((z - center) ** 2, dim=1)  # Squared distance from the center
            scores.extend(dist.numpy())
            labels.extend(y.numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    auc = roc_auc_score(labels, -scores)  # Negative distance to center
    return auc

# Step 6: Hyperparameter Grid Search
def perform_grid_search(X_train, X_test, y_train, y_test):
    """
    Perform grid search over hidden dimensions, learning rates, and epochs.
    """
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    hidden_dim_grid = [16, 32, 64]
    lr_grid = [1e-3, 1e-4]
    epoch_grid = [20, 30, 50]

    best_auc = 0
    best_params = None

    for hidden_dim, lr, n_epochs in itertools.product(hidden_dim_grid, lr_grid, epoch_grid):
        # Initialize the model and center
        model = DeepSVDD(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
        center = initialize_center(model, train_loader)

        train_deep_svdd(model, center, train_loader, n_epochs=n_epochs, lr=lr)
        auc = evaluate_deep_svdd(model, center, test_loader)

        if auc > best_auc:
            best_auc = auc
            best_params = {"hidden_dim": hidden_dim, "lr": lr, "n_epochs": n_epochs}

    print("Best AUC-ROC:", best_auc)
    print("Best Parameters:", best_params)

# Step 7: Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    perform_grid_search(X_train, X_test, y_train, y_test)
