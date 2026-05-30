# Run this script once to regenerate your test CSV with a proper fault ramp
import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\lenovo\\Documents\\Thesis\\thesis\\GCL\\Thesis_2\\GCL_V1\\data\\raw\\test1.csv")

# Find the last N rows — this is where your anomaly should build up
ramp_start = 200   # start of fault development (row index)
ramp_end   = 324   # end of file

n_ramp = ramp_end - ramp_start  # 124 rows of gradual fault

# Gradually increase voltage imbalance and temperature — realistic fault behaviour
for i, row_idx in enumerate(range(ramp_start, ramp_end)):
    progress = i / n_ramp                    # 0.0 → 1.0

    # Voltage imbalance grows gradually

    # Temperature rises gradually (most realistic fault signal)
    df.loc[row_idx, 'motor_temp_C'] += 20 * progress

    # Power factor degrades

df.to_csv("test2.csv", index=False)
print("Done — fault ramp injected over rows 200–324")