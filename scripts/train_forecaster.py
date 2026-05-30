# scripts/train_forecaster.py
import pandas as pd
import numpy as np
from src.utils.error_forecaster import train_forecaster

# Step 1: load healthy validation errors (what you already have)
healthy = pd.read_csv("results/global_reconstruction_error.csv")
healthy_errors = healthy["global_reconstruction_error"].values.tolist()

# Step 2: run inference.py first to get test errors, then load them
# (after running inference once with the fix above, save them like this)
# We'll build a combined training sequence: healthy → rising → fault
# This teaches the LSTM what a fault build-up looks like

test_errors = np.loadtxt("results/test_errors.csv", delimiter=",").tolist()
combined_errors = healthy_errors + test_errors

print(f"Training LSTM on {len(combined_errors)} error samples "
      f"({len(healthy_errors)} healthy + {len(test_errors)} test errors)")

train_forecaster(combined_errors, seq_len=20, forecast_steps=10, epochs=300)