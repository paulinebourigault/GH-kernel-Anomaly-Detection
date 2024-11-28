# Kernel-Based Anomaly Detection Using Generalized Hyperbolic Processes

## **GH-Kernel-Anomaly-Detection**

This repository contains implementations of various anomaly detection algorithms and methods. It includes deep learning models, kernel-based approaches, and traditional methods for anomaly detection on synthetic datasets. The focus is on flexibility, ease of use, and reproducibility.

## **Repository Structure**

The repository includes the following Python scripts:

| File                               | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `dagmm_anomaly_detection.py`       | Implements Deep Autoencoding Gaussian Mixture Model (DAGMM) for anomaly detection. |
| `deep_svdd_anomaly_detection.py`   | Implements Deep Support Vector Data Description (Deep SVDD).               |
| `isolation_forest_anomaly_detection.py` | Implements Isolation Forest with hyperparameter tuning.                     |
| `KDE_GH_standard_kernels.py`       | Implements KDE using Generalized Hyperbolic (GH) kernels and standard kernels. |
| `memae_anomaly_detection.py`       | Implements Memory-Augmented Autoencoder (MemAE) for anomaly detection.     |
| `ocnn_anomaly_detection.py`        | Implements One-Class Neural Network (OC-NN) for anomaly detection.         |
| `OCSVM_GH_standard_kernels.py`     | Implements One-Class SVM with GH and standard kernels.                     |
| `vae_anomaly_detection.py`         | Implements Variational Autoencoder (VAE) for anomaly detection.            |
| `vanilla_autoencoder_anomaly_detection.py` | Implements a simple Autoencoder for anomaly detection.                   |

## **Requirements**

To run the scripts, you need the following Python libraries:

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- pandas
- tqdm (optional, for progress tracking)

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt

## **Datasets**
- synthetic
- KDDCup99
- ForestCover

## **Usage**

Each script is self-contained and can be run independently. For example, to run the DAGMM implementation:

```bash
python dagmm_anomaly_detection.py

## **References**
To be added.

