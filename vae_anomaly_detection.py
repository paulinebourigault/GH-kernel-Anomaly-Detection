"""
Variational Autoencoder (VAE) for Anomaly Detection

This script implements a Variational Autoencoder (VAE) to detect anomalies in a synthetic non-Gaussian dataset.

Author: Pauline Bourigault
Date: 27/11/2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import itertools


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


# Step 2: Model Definition
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model with customizable hidden and latent dimensions.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # latent_dim * 2 for mean and logvar
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        params = self.encoder(x)
        mean, logvar = torch.chunk(params, 2, dim=1)  # Split the output into mean and logvar
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar


# Step 3: Loss Function
def vae_loss_function(reconstructed, original, mean, logvar):
    """
    Compute the loss for the Variational Autoencoder.
    Combines reconstruction loss and KL divergence.
    """
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence / original.size(0) 


# Step 4: Training and Evaluation
def train_vae(X_train, X_test, y_test, hidden_dims, latent_dims, learning_rates, num_epochs):
    """
    Train and evaluate the Variational Autoencoder using grid search over hyperparameters.
    Returns:
        best_auc: Best AUC-ROC score
        best_params: Parameters corresponding to the best AUC-ROC
        results_df: DataFrame containing results for all configurations
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_auc = 0
    best_params = {}
    results = []

    for hidden_dim, latent_dim, lr in itertools.product(hidden_dims, latent_dims, learning_rates):
        # Initialize the model
        model = VAE(input_dim=X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                batch_data = batch[0]
                reconstructed, mean, logvar = model(batch_data)
                loss = vae_loss_function(reconstructed, batch_data, mean, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            reconstructed_test, _, _ = model(X_test_tensor)
            reconstructed_test = reconstructed_test.numpy()
            reconstruction_error = ((X_test_tensor.numpy() - reconstructed_test) ** 2).sum(axis=1)

            auc = roc_auc_score(y_test, reconstruction_error)

            if auc > best_auc:
                best_auc = auc
                best_params = {"hidden_dim": hidden_dim, "latent_dim": latent_dim, "learning_rate": lr}

            results.append({
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "learning_rate": lr,
                "AUC-ROC": auc
            })

    results_df = pd.DataFrame(results)
    return best_auc, best_params, results_df


# Step 5: Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()

    hidden_dims = [16, 32, 64]
    latent_dims = [2, 4, 6, 8]
    learning_rates = [0.001, 0.01]
    num_epochs = [20, 30, 50]

    best_auc, best_params, results_df = train_vae(
        X_train, X_test, y_test, hidden_dims, latent_dims, learning_rates, num_epochs
    )

    print("Best AUC-ROC:", best_auc)
    print("Best Parameters:", best_params)
    print(results_df)

    results_df.to_csv("vae_anomaly_results.csv", index=False)
