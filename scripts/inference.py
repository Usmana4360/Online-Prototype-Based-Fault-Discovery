import torch
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.config import *
from src.lightning.gcl_module import GCLConv1DUnsupervised
from src.datasets.sensor_dataset import SensorDataset
import numpy as np

from src.utils.reconstruction import feature_contribution
from src.utils.fault_signature import (
    create_signature,
    extract_anomaly_signatures,
    cluster_signatures,
    save_fault_library,
    classify_new_signature
)
from src.utils.visualization import plot_signature_pca

def load_model(checkpoint_path, scaler, test_dataset=None):
    model = GCLConv1DUnsupervised(
        scaler=scaler,
        feature_cols=FEATURE_COLS,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        latent_channels=LATENT_CHANNELS,
        test_dataset=test_dataset
    )

    # PyTorch 2.6+: weights_only=False
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

def run_inference(model, df, scaler):
    dataset = SensorDataset(df, scaler, FEATURE_COLS, CLIP_LEN, STRIDE, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_global_errors = []
    all_per_feature_errors = []
    all_contributions = []


    with torch.no_grad():
        for batch in loader:
            x = batch
            x_hat, _ = model(x)

            global_err = torch.mean((x - x_hat) ** 2, dim=(1, 2))
            per_feat_err = torch.mean((x - x_hat) ** 2, dim=1)
            contrib = per_feat_err / (per_feat_err.sum(dim=1, keepdim=True) + 1e-8)

            all_global_errors.append(global_err.item())
            all_per_feature_errors.append(per_feat_err.squeeze().numpy())
            all_contributions.append(contrib.squeeze().numpy())

    return all_global_errors, all_per_feature_errors, all_contributions

def visualize_results(global_errors, per_feature_errors):
    plt.figure(figsize=(6,4))
    plt.plot(global_errors)
    plt.axhline(np.mean(global_errors), linestyle='--')
    plt.title("Global Reconstruction Error (All Windows)")
    plt.ylabel("MSE")
    plt.xlabel("Window Index")
    plt.show()

    per_feature_errors = pd.DataFrame(per_feature_errors, columns=FEATURE_COLS)
    plt.figure()
    per_feature_errors.mean().plot(kind='bar')
    plt.title("Average Per-Feature Reconstruction Error")
    plt.show()

def main():
    df = pd.read_csv("data/raw/test1.csv")
    scaler = joblib.load("scalers/scaler.save")

    dataset = SensorDataset(df, scaler, FEATURE_COLS, CLIP_LEN, STRIDE, split="test")

    checkpoint_path = "checkpoints/gcl-epoch=39-val_recon_loss=0.0000.ckpt"
    model = load_model(checkpoint_path, scaler, test_dataset=dataset)

    global_errors, per_feature_errors, contributions = run_inference(model, df, scaler)

    # -------------------------------------------------
    # 🔥 STEP 1: Extract anomaly signatures
    # -------------------------------------------------
    healthy_errors = pd.read_csv("results/global_reconstruction_error.csv")
    healthy_errors = healthy_errors["global_reconstruction_error"].values

    mu = np.mean(healthy_errors)
    sigma = np.std(healthy_errors)
    threshold = mu + 3 * sigma

    print(f"Anomaly Threshold: {threshold}")

    signatures, indices = [], []

    for idx, (g, c) in enumerate(zip(global_errors, contributions)):
        if g > threshold:
            signatures.append(create_signature(c, g))
            indices.append(idx)

    print(f"Total Anomalies Detected: {len(signatures)}")
    # =====================================================
    # 🔷 CASE 1: If multiple anomalies → cluster + save
    # =====================================================
    if len(signatures) > 1:

        # STEP 2: Cluster signatures
        labels, centroids, sil_score = cluster_signatures(signatures)

        print("Clusters discovered:", len(centroids))
        print("Silhouette Score:", sil_score)

        # STEP 3: Save Fault Prototype Library
        save_fault_library(centroids, sil_score)
        print("Fault Prototype Library Saved.")

    else:
        print("Not enough anomalies to cluster.")

    # =====================================================
    # 🔷 STEP 4: CLASSIFY USING EXISTING LIBRARY
    # =====================================================
    try:
        print("\n--- Fault Classification Results ---")

        for i, sig in enumerate(signatures):
            result = classify_new_signature(sig)

            print(f"Anomaly {indices[i]} → {result}")

    except Exception as e:
        print("Fault library not found or classification error:", e)

    visualize_results(global_errors, per_feature_errors)
if __name__ == "__main__":
    main()
