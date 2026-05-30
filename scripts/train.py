import os
import pandas as pd
import joblib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config import *
from src.lightning.gcl_module import GCLConv1DUnsupervised
from src.datasets.sensor_dataset import SensorDataset
from src.utils.preprocessing import fit_scaler_on_train
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
pl.seed_everything(SEED, workers=True)
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("scalers", exist_ok=True)

    # Load data
    df = pd.read_csv("data/raw/55kw_motor_data.csv")

    # Fit scaler
    scaler = fit_scaler_on_train(df, FEATURE_COLS, CLIP_LEN, STRIDE)
    joblib.dump(scaler, "scalers/scaler.save")

    # Create datasets
    train_ds = SensorDataset(df, scaler, FEATURE_COLS, CLIP_LEN, STRIDE, split="train")
    val_ds = SensorDataset(df, scaler, FEATURE_COLS, CLIP_LEN, STRIDE, split="val")

    # Instantiate model
    model = GCLConv1DUnsupervised(
        scaler=scaler,
        feature_cols=FEATURE_COLS,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        lr=LR,
        latent_channels=LATENT_CHANNELS,
        train_dataset=train_ds,
        val_dataset=val_ds
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="gcl-{epoch:02d}-{val_recon_loss:.4f}",
        monitor="val/recon_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    trainer = pl.Trainer(
        max_epochs=40,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )

    trainer.fit(model)
    print("✅ Training complete! Best model saved at:", checkpoint_callback.best_model_path)

if __name__ == "__main__":
    main()
