import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Predict race time or position
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history

def prepare_driver_sequence(df, driver_name, feature_cols, max_races, scaler):
    df = df[df["Name"] == driver_name].sort_values("Round")

    if len(df) == 0:
        return None

    features = df[feature_cols].values

    # Pad/trim to match model input length
    if features.shape[0] > max_races:
        features = features[-max_races:]
    elif features.shape[0] < max_races:
        padding = np.zeros((max_races - features.shape[0], features.shape[1]))
        features = np.vstack([padding, features])

    features = features[:-1]  # exclude current race
    features = scaler.transform(features)  # apply normalization
    return features[np.newaxis, ...]  # add batch dimension