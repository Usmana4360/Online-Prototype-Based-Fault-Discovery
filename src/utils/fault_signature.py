import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import pickle
import os


# -------------------------------------------------
# 1️⃣ Create Fault Signature Vector
# (Using feature contribution for stability)
# -------------------------------------------------
def create_signature(contribution_vector, global_error=None):

    vec = np.array(contribution_vector)

    # Direction component
    direction = vec / (np.linalg.norm(vec) + 1e-8)

    if global_error is None:
        return direction

    # Magnitude component (severity)
    severity = np.array([global_error])

    # Combine direction + magnitude
    signature = np.concatenate([direction, severity])

    return signature


# -------------------------------------------------
# 2️⃣ Extract anomaly signatures
# -------------------------------------------------
def extract_anomaly_signatures(global_errors,
                               contributions,
                               percentile_threshold=95):

    threshold = np.percentile(global_errors, percentile_threshold)

    signatures = []
    indices = []

    for idx, (g, c) in enumerate(zip(global_errors, contributions)):
        if g > threshold:
            signatures.append(create_signature(c))
            indices.append(idx)

    return np.array(signatures), indices, threshold


# -------------------------------------------------
# 3️⃣ Cluster Fault Signatures
# -------------------------------------------------
def cluster_signatures(signatures, max_k=6):

    signatures = np.array(signatures)

    n_samples = len(signatures)

    # Not enough data
    if n_samples < 2:
        return None, None, None

    best_score = -1
    best_labels = None
    best_kmeans = None

    # k must be < n_samples
    max_allowed_k = min(max_k, n_samples - 1)

    for k in range(2, max_allowed_k + 1):

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(signatures)

        score = silhouette_score(signatures, labels, metric="cosine")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_kmeans = kmeans

    return best_labels, best_kmeans.cluster_centers_, best_score


# -------------------------------------------------
# 4️⃣ Save Fault Prototype Library
# -------------------------------------------------
def save_fault_library(centroids,
                       silhouette_score=None,
                       path="results/fault_prototype_library.pkl"):

    fault_library = {
        "centroids": np.array(centroids),
        "num_clusters": len(centroids),
        "silhouette_score": silhouette_score
    }

    with open(path, "wb") as f:
        pickle.dump(fault_library, f)

    print("✅ Fault library saved correctly at:", path)


def load_fault_library(path="results/fault_prototype_library.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_fault_library(path="results/fault_prototype_library.pkl"):
    return joblib.load(path)


# -------------------------------------------------
# 5️⃣ Real-Time Classification
# -------------------------------------------------
def classify_new_signature(signature,
                           library_path="results/fault_prototype_library.pkl",
                           similarity_threshold=0.75):

    if not os.path.exists(library_path):
        return {"status": "Library Not Found"}

    # Normalize
    signature = signature / (np.linalg.norm(signature) + 1e-8)

    fault_lib = load_fault_library(library_path)
    centroids = fault_lib["centroids"]

    similarities = cosine_similarity(
        signature.reshape(1, -1),
        centroids
    )

    best_idx = np.argmax(similarities)
    best_score = similarities[0, best_idx]

    if best_score >= similarity_threshold:
        return {
            "status": "Known Fault",
            "cluster": int(best_idx),
            "confidence": float(best_score)
        }
    else:
        return {
            "status": "Unknown Fault",
            "cluster": None,
            "confidence": float(best_score)
        }