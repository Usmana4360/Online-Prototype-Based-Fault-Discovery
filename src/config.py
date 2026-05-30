FEATURE_COLS = [
    'volt_V1','volt_V2','volt_V3',
    'current_A1','current_A2','current_A3',
    'power_kW','power_factor','motor_temp_C'
]

CLIP_LEN = 100
STRIDE = 10
BATCH_SIZE = 64
LR = 1e-4
LATENT_CHANNELS = 16
INCLUDE_SEVERITY_IN_SIGNATURE = True