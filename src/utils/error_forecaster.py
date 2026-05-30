# src/utils/error_forecaster.py
import numpy as np
import torch
import torch.nn as nn

class ErrorLSTM(nn.Module):
    """
    A small LSTM that takes the last `seq_len` error values
    and predicts the next `forecast_steps` values.
    """
    def __init__(self, seq_len=30, hidden=32, forecast_steps=10):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, forecast_steps)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # predict from last hidden state


def train_forecaster(error_history, seq_len=30, forecast_steps=10,
                     epochs=100, lr=1e-3):
    """
    Train the LSTM on the recorded reconstruction error history.
    Call this ONCE after you have enough error data (e.g. from validation).

    error_history: list or array of past reconstruction errors
    """
    errors = np.array(error_history, dtype=np.float32)

    # Normalize
    mu, sigma = errors.mean(), errors.std() + 1e-8
    errors_norm = (errors - mu) / sigma

    # Build sequences
    X, Y = [], []
    for i in range(len(errors_norm) - seq_len - forecast_steps):
        X.append(errors_norm[i:i+seq_len])
        Y.append(errors_norm[i+seq_len:i+seq_len+forecast_steps])

    X = torch.tensor(np.array(X)).unsqueeze(-1)  # (N, seq_len, 1)
    Y = torch.tensor(np.array(Y))                 # (N, forecast_steps)

    model = ErrorLSTM(seq_len=seq_len, forecast_steps=forecast_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Save normalisation stats with the model
    torch.save({"state_dict": model.state_dict(),
                "mu": mu, "sigma": sigma,
                "seq_len": seq_len,
                "forecast_steps": forecast_steps},
               "results/error_forecaster.pt")
    print("✅ Forecaster saved to results/error_forecaster.pt")
    return model, mu, sigma


def predict_future_errors(error_history,
                          model_path="results/error_forecaster.pt"):
    """
    Load the trained LSTM and predict the next N error values.
    Returns a numpy array of predicted future errors (original scale).
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    seq_len    = checkpoint["seq_len"]
    mu         = checkpoint["mu"]
    sigma      = checkpoint["sigma"]

    model = ErrorLSTM(seq_len=seq_len,
                      forecast_steps=checkpoint["forecast_steps"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # ── FIX: handle case where history is shorter than seq_len ──
    if len(error_history) < seq_len:
        # Pad with healthy mean on the left
        pad_len = seq_len - len(error_history)
        padding = [float(mu)] * pad_len
        errors  = np.array(padding + list(error_history), dtype=np.float32)
        print(f"[Forecaster] padded {pad_len} values (history too short for seq_len={seq_len})")
    else:
        errors = np.array(error_history[-seq_len:], dtype=np.float32)

    errors_norm = (errors - mu) / (sigma + 1e-8)
    x = torch.tensor(errors_norm).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

    with torch.no_grad():
        pred_norm = model(x).squeeze().numpy()

    return pred_norm * sigma + mu  # back to original scale