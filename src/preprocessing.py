import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_lstm_sequences(
    df: pd.DataFrame,
    feature_cols=None,
    target_col="BestLapTime",
    min_races=2,
    max_races=None
):
    """
    Convert driver-race data into padded LSTM sequences.
    Returns:
        X: np.array of shape (num_drivers, max_races - 1, num_features)
        y: np.array of shape (num_drivers,) — target for last race
        names: list of driver names
    """
    if feature_cols is None:
        feature_cols = [
            "BestLapTime", "AvgLapTime", "PitStops", "QualifyingTime",
            "TeamAvgBestLapTime", "TeamAvgLapTime", "TeamAvgQualiTime"
        ]

    grouped = df.groupby("Name")
    X, y, names = [], [], []

    # Determine max number of races (unless explicitly set)
    if max_races is None:
        max_races = df["Round"].nunique()

    num_features = len(feature_cols)

    for name, group in grouped:
        group = group.sort_values("Round")

        if len(group) < min_races:
            continue

        # Drop rows with missing required features
        if group[feature_cols + [target_col]].isna().any().any():
            continue

        features = group[feature_cols].values
        target = group[target_col].values[-1]

        # Trim if too long
        if features.shape[0] > max_races:
            features = features[-max_races:]
        # Pad if too short (pre-pad with zeros)
        elif features.shape[0] < max_races:
            padding = np.zeros((max_races - features.shape[0], num_features))
            features = np.vstack([padding, features])

        X.append(features[:-1])  # exclude last race for input
        y.append(target)         # last race is the label
        names.append(name)

    return np.array(X), np.array(y), names

def normalize_lstm_data(X: np.ndarray, y: np.ndarray):

    num_samples, timesteps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_reshaped).reshape(num_samples, timesteps, num_features)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, x_scaler, y_scaler