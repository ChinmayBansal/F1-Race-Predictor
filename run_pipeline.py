import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model_lstm import train_lstm_model
from src.data_collection import get_race_data
from src.data_collection import get_race_data
from src.preprocessing import create_lstm_sequences
from src.preprocessing import normalize_lstm_data
from src.model_lstm import train_lstm_model, prepare_driver_sequence
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# -------- Settings -------- #
train_rounds = [1, 2, 3, 4]
predict_round = 5
feature_cols = [
    "BestLapTime", "AvgLapTime", "PitStops", "QualifyingTime",
    "TeamAvgBestLapTime", "TeamAvgLapTime", "TeamAvgQualiTime"
]
max_races = len(train_rounds) + 1

# -------- Load Data -------- #
print("📦 Loading race data...")
dfs = [get_race_data(2025, r) for r in train_rounds + [predict_round]]
df_all = pd.concat(dfs, ignore_index=True)

train_df = df_all[df_all["Round"].isin(train_rounds)]
predict_df = df_all[df_all["Round"] <= predict_round]

# -------- Preprocessing -------- #
print("⚙️ Preprocessing sequences...")
X, y, names = create_lstm_sequences(train_df, feature_cols, "BestLapTime", min_races=2, max_races=max_races)
X_norm, y_norm, x_scaler, y_scaler = normalize_lstm_data(X, y)

# -------- Train LSTM -------- #
print("🧠 Training LSTM...")
model, _ = train_lstm_model(X_norm, y_norm, X_norm[-4:], y_norm[-4:])
model.save("models/lstm_model.h5")

# -------- Predict Round 5 (Saudi Arabia) -------- #
print("\n🏁 Predicted Best Lap Times for 2025 Saudi Arabian GP:\n")

predictions = []

for name in sorted(predict_df["Name"].unique()):
    input_seq = prepare_driver_sequence(predict_df, name, feature_cols, max_races, x_scaler)
    if input_seq is None:
        continue

    pred_norm = model.predict(input_seq).flatten()[0]
    pred_real = y_scaler.inverse_transform([[pred_norm]])[0][0]
    predictions.append((name, pred_real))

# -------- Rank Results -------- #
predictions.sort(key=lambda x: x[1])
print("\n📊 Ranked Predictions:")
for i, (name, time) in enumerate(predictions, 1):
    print(f"{i:>2}. {name:20s} → {time:.3f} sec")

# Compare to actuals (if available)
actuals = predict_df[predict_df["Round"] == predict_round][["Name", "BestLapTime"]].dropna()

# Join predicted + actual
results_df = pd.DataFrame(predictions, columns=["Name", "PredictedTime"])
results_df = results_df.merge(actuals, on="Name", how="inner")

# Calculate metrics
y_true = results_df["BestLapTime"].values
y_pred = results_df["PredictedTime"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n✅ Model Evaluation on 2025 Saudi Arabian GP:")
print(f"Mean Absolute Error (MAE):     {mae:.3f} sec")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} sec")
print(f"R² Score:                        {r2:.3f}")

# -------- Optional Plot -------- #
names_sorted, times_sorted = zip(*predictions)
plt.figure(figsize=(10, 6))
plt.barh(names_sorted[::-1], times_sorted[::-1])
plt.title("Predicted Best Lap Times – 2025 Saudi GP")
plt.xlabel("Best Lap Time (s)")
plt.grid(True)
plt.tight_layout()
plt.show()