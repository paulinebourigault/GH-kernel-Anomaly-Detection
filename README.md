# Kernel-Based Anomaly Detection Using Generalized Hyperbolic Processes

We present a novel approach to anomaly detection by integrating Generalized Hyperbolic (GH) processes into kernel-based methods. The GH distribution, known for its flexibility in modeling skewness, heavy tails, and kurtosis, helps to capture complex patterns in data that deviate from Gaussian assumptions. We propose a GH-process-based kernel function and utilize it within kernel density estimation (KDE) and One- Class Support Vector Machines (OCSVM) to develop robust anomaly detection frameworks. Theoretical results confirm the positive semi-definiteness and consistency of the GH-based kernel, ensuring its suitability for machine learning applications.

<p align="center" width="100%">
    <img src=img/workflow-anomalydetection.png width="25%"/>
</p>

## **Repository Structure**

The repository includes the following Python scripts:

| File                               | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `KDE_GH_standard_kernels.py`       | Implements KDE using Generalized Hyperbolic (GH) kernels (Full GH Kernel, Gaussian, NIG, Student's t, Hyperbolic) and standard kernels (Gaussian, Tophat, Exponential, Epanechnikov). |
| `OCSVM_GH_standard_kernels.py`     | Implements One-Class SVM with GH (Full GH Kernel, Gaussian, NIG, Student's t, Hyperbolic) and standard (RBF, Polynomial, Linear, Sigmoid) kernels.                     |
| `dagmm_anomaly_detection.py`       | Implements Deep Autoencoding Gaussian Mixture Model (DAGMM) for anomaly detection. |
| `deep_svdd_anomaly_detection.py`   | Implements Deep Support Vector Data Description (Deep SVDD).               |
| `isolation_forest_anomaly_detection.py` | Implements Isolation Forest for anomaly detection.                     |
| `memae_anomaly_detection.py`       | Implements Memory-Augmented Autoencoder (MemAE) for anomaly detection.     |
| `ocnn_anomaly_detection.py`        | Implements One-Class Neural Network (OC-NN) for anomaly detection.         |
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
```

## **Datasets**

- synthetic
- KDDCup99
- ForestCover

## **Usage**

Each script is self-contained and can be run independently. For example, to run the KDE using Generalized Hyperbolic (GH) kernels (Full GH Kernel, Gaussian, NIG, Student's t, Hyperbolic) and standard kernels (Gaussian, Tophat, Exponential, Epanechnikov) implementation:

```bash
python KDE_GH_standard_kernels.py
```

## **References**

To be added.
