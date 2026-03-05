from sklearn.preprocessing import StandardScaler

def fit_scaler_on_train(df, feature_cols, clip_len, stride):
    L = len(df)
    starts = list(range(0, L - clip_len + 1, stride))
    n_train = int(0.7 * len(starts))
    last_start = starts[n_train - 1]
    train_rows = df.iloc[:last_start + clip_len]

    scaler = StandardScaler()
    scaler.fit(train_rows[feature_cols].values)
    return scaler
