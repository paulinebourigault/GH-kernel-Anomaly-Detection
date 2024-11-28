"""
Deep Autoencoding Gaussian Mixture Model (DAGMM) for Anomaly Detection

This script implements the DAGMM framework for anomaly detection on a synthetic non-Gaussian dataset.
The model combines autoencoding with a Gaussian Mixture Model (GMM) to compute anomaly scores.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import itertools
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
    y = np.where(y == 0, 1, -1)  # Map classes: 1 (normal), -1 (anomalies)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Step 2: DAGMM Model Definition
class DAGMM(nn.Module):
    """
    Deep Autoencoding Gaussian Mixture Model (DAGMM)
    Combines an autoencoder with a Gaussian Mixture Model for anomaly detection.
    """
    def __init__(self, input_dim, latent_dim):
        super(DAGMM, self).__init__()
        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim)
        )
        # Estimation network
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 1, 10),
            nn.Tanh(),
            nn.Linear(10, 2),  # Number of GMM components
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        rec_error = torch.mean((x - x_hat) ** 2, dim=1, keepdim=True)  # Reconstruction error
        z = torch.cat([z_c, rec_error], dim=1)  # Latent + reconstruction error
        gamma = self.estimation(z)  # Softmax probabilities
        return z_c, x_hat, rec_error, z, gamma

# Step 3: Grid Search for Hyperparameter Tuning
def perform_grid_search(X_train, X_test, y_train, y_test):
    """
    Perform grid search over latent dimensions, learning rates, GMM components, and batch sizes.
    """
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    # Hyperparameter grid
    latent_dims = [2, 3, 5]
    learning_rates = [1e-3, 1e-4]
    n_components_grid = [2, 3]
    batch_sizes = [16, 32]

    # Grid search results
    best_auc = 0
    best_params = {}
    results = []

    # Perform grid search
    for latent_dim, lr, n_components, batch_size in itertools.product(latent_dims, learning_rates, n_components_grid, batch_sizes):
        # Update DataLoader for batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize DAGMM model
        model = DAGMM(input_dim=X_train.shape[1], latent_dim=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Training phase
        epochs = 50
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x, _ = batch
                optimizer.zero_grad()
                z_c, x_hat, rec_error, _, _ = model(x)
                recon_loss = loss_fn(x_hat, x)
                recon_loss.backward()
                optimizer.step()
                train_loss += recon_loss.item()

        # Extract latent representations for GMM fitting
        model.eval()
        z_train = []
        for batch in train_loader:
            x, _ = batch
            z_c, _, rec_error, _, _ = model(x)
            z = torch.cat([z_c, rec_error], dim=1).detach().numpy()
            z_train.append(z)
        z_train = np.concatenate(z_train, axis=0)

        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
        gmm.fit(z_train)

        # Testing phase
        test_scores = []
        for batch in test_loader:
            x, _ = batch
            z_c, _, rec_error, _, _ = model(x)
            z = torch.cat([z_c, rec_error], dim=1).detach().numpy()
            score = -gmm.score_samples(z)  # Negative log-likelihood
            test_scores.extend(score)

        # Evaluation metrics
        auc = roc_auc_score(y_test, -np.array(test_scores))

        # Update best results
        if auc > best_auc:
            best_auc = auc
            best_params = {
                "latent_dim": latent_dim,
                "learning_rate": lr,
                "n_components": n_components,
                "batch_size": batch_size,
            }

        results.append({
            "latent_dim": latent_dim,
            "learning_rate": lr,
            "n_components": n_components,
            "batch_size": batch_size,
            "AUC-ROC": auc,
        })

    results_df = pd.DataFrame(results)

    # Output the best results
    print("Best AUC-ROC:", best_auc)
    print("Best Parameters:", best_params)
    print(results_df)
    return results_df

# Step 4: Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    results_df = perform_grid_search(X_train, X_test, y_train, y_test)
