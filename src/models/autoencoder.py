import torch.nn as nn

class SimpleConv1DAutoencoder(nn.Module):
    def __init__(self, n_features, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_channels, 3, padding=1),
            nn.BatchNorm1d(latent_channels),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, n_features, 3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
