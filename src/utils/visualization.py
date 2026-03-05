import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_signature_pca(signatures, labels):
    """
    signatures: numpy array (N_anomalies, n_features)
    labels: cluster labels from clustering
    """

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(signatures)

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)

    for lab in unique_labels:
        idx = labels == lab
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            label=f"Cluster {lab}",
            s=80
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Fault Signature Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Explained variance ratio:", pca.explained_variance_ratio_)
