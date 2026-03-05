import torch

def global_reconstruction_error(x, x_hat):
    # mean over time and features
    return torch.mean((x - x_hat) ** 2, dim=(1, 2))


def per_feature_reconstruction_error(x, x_hat):
    # mean over time dimension only
    # x shape: (B, clip_len, n_features)
    sq_err = (x - x_hat) ** 2
    return torch.mean(sq_err, dim=1)   # (B, n_features)


def feature_contribution(per_feature_error):
    # normalize across features
    return per_feature_error / (
        per_feature_error.sum(dim=1, keepdim=True) + 1e-8
    )
