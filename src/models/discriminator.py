import torch.nn as nn

class SimpleConv1DDiscriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.conv(x).mean(dim=-1)
        return self.classifier(h)
