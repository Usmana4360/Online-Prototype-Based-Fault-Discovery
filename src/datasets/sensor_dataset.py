import numpy as np
import torch
from torch.utils.data import Dataset

class SensorDataset(Dataset):
    def __init__(self, df, scaler, feature_cols, clip_len, stride, split='train'):
        X = scaler.transform(df[feature_cols].values.astype(np.float32))

        windows = []
        for i in range(0, len(X) - clip_len + 1, stride):
            windows.append(X[i:i+clip_len])

        windows = np.stack(windows)

        n = len(windows)
        n_train = int(0.7 * n)
        n_val = int(0.85 * n)

        if split == 'train':
            windows = windows[:n_train]
        elif split == 'val':
            windows = windows[n_train:n_val]
        else:
            windows = windows[n_val:]

        self.data = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
