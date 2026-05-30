# src/lightning/gcl_module.py
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader

from src.models.autoencoder import SimpleConv1DAutoencoder
from src.models.discriminator import SimpleConv1DDiscriminator
from src.datasets.sensor_dataset import SensorDataset
from src.utils.reconstruction import (
    global_reconstruction_error,
    per_feature_reconstruction_error,
    feature_contribution
)

class GCLConv1DUnsupervised(pl.LightningModule):

    def __init__(
        self,
        scaler,
        feature_cols,
        clip_len=100,
        batch_size=32,
        lr=1e-4,
        latent_channels=16,
        stride=10,
        num_workers=2,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["scaler", "train_dataset", "val_dataset", "test_dataset"])

        self.scaler = scaler
        self.feature_cols = feature_cols
        self.clip_len = clip_len
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        n_features = len(self.feature_cols)

        self.generator = SimpleConv1DAutoencoder(
            n_features=n_features,
            latent_channels=latent_channels
        )
        self.discriminator = SimpleConv1DDiscriminator(n_features=n_features)

        self.mse = nn.MSELoss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.validation_outputs = []

    # ---------------- Forward ----------------
    def forward(self, x):
        x_conv = x.permute(0, 2, 1)  # (B, n_features, clip_len)
        x_hat_conv, z = self.generator(x_conv)
        x_hat = x_hat_conv.permute(0, 2, 1)  # (B, clip_len, n_features)
        return x_hat, z

    # ---------------- Training Step ----------------
    def training_step(self, batch, batch_idx):
        x = batch
        device = x.device
        opt_d, opt_g = self.optimizers()

        x_conv = x.permute(0, 2, 1)

        # ----- Discriminator -----
        x_hat_conv, _ = self.generator(x_conv)
        real_logits = self.discriminator(x_conv)
        fake_logits = self.discriminator(x_hat_conv.detach())

        real_labels = torch.ones_like(real_logits, device=device)
        fake_labels = torch.zeros_like(fake_logits, device=device)

        d_loss_real = self.bce_logits(real_logits, real_labels)
        d_loss_fake = self.bce_logits(fake_logits, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # ----- Generator -----
        x_hat_conv, _ = self.generator(x_conv)
        fake_logits = self.discriminator(x_hat_conv)
        adv_loss = self.bce_logits(fake_logits, torch.ones_like(fake_logits, device=device))

        x_hat = x_hat_conv.permute(0, 2, 1)
        recon_loss = self.mse(x_hat, x)
        g_loss = recon_loss + 1e-3 * adv_loss

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({
            "train/d_loss": d_loss,
            "train/g_loss": g_loss,
            "train/recon_loss": recon_loss
        }, prog_bar=True)

    # ---------------- Validation Step ----------------
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, _ = self(x)
        global_err = global_reconstruction_error(x, x_hat)
        per_feat_err = per_feature_reconstruction_error(x, x_hat)
        contrib = feature_contribution(per_feat_err)

        self.validation_outputs.append({
            "global_error": global_err.detach().cpu(),
            "per_feature_error": per_feat_err.detach().cpu(),
            "contribution": contrib.detach().cpu()
        })
        self.log("val/recon_loss", global_err.mean(), prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return

        os.makedirs("results", exist_ok=True)

        all_global = torch.cat([o["global_error"] for o in self.validation_outputs]).numpy()
        all_per_feat = torch.cat([o["per_feature_error"] for o in self.validation_outputs]).numpy()
        all_contrib = torch.cat([o["contribution"] for o in self.validation_outputs]).numpy()

        pd.DataFrame({"global_reconstruction_error": all_global}).to_csv(
            "results/global_reconstruction_error.csv", index=False
        )
        pd.DataFrame(all_per_feat, columns=self.feature_cols).to_csv(
            "results/per_feature_reconstruction_error.csv", index=False
        )
        pd.DataFrame(all_contrib, columns=self.feature_cols).to_csv(
            "results/feature_contribution.csv", index=False
        )
        print("✅ Validation results saved in results/ folder")
        self.validation_outputs.clear()

    # ---------------- Optimizers ----------------
    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        return [opt_d, opt_g]
    
    def test_step(self, batch, batch_idx):
        x = batch
        x_hat, z = self(x)
        global_err = global_reconstruction_error(x, x_hat)
        per_feat_err = per_feature_reconstruction_error(x, x_hat)
        contrib = feature_contribution(per_feat_err)
        self.log("test/recon_loss", global_err.mean(), prog_bar=True)
        return {
            "global_error": global_err.detach().cpu(),
            "per_feature_error": per_feat_err.detach().cpu(),
            "contribution": contrib.detach().cpu(),
        }

    # ---------------- DataLoaders ----------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
