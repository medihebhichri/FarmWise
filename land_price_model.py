import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# === 1. Load model & scalers ===
model = load_model("land_price_lstm_model.h5", compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
feat_scaler = joblib.load("feat_scaler_lstm.pkl")
target_scaler = joblib.load("target_scaler_lstm.pkl")

# === 2. Load training structure for one-hot encoding ===
df_train = pd.read_csv("Modified_Tunisian_Land_Pricing_Dataset_v2.csv")
df_train["Log Area"] = np.log1p(df_train["Total Land (m²)"])
features = ["Year", "Log Area", "Region", "Region or State"]
df_train_encoded = pd.get_dummies(df_train[features])

# === 3. User Input ===
print("📍 Predict Land Price per m² (TND)")
year = int(input("Enter Year (e.g. 2025): "))
area = float(input("Enter Total Land Area (m²): "))
region = input("Enter Region (North / Center / South): ")
state = input("Enter Governorate (e.g. Tunis, Sfax): ")

# === 4. Preprocess input ===
input_data = {
    "Year": year,
    "Total Land (m²)": area,
    "Region": region,
    "Region or State": state
}
input_data["Log Area"] = np.log1p(input_data["Total Land (m²)"])
input_df = pd.DataFrame([input_data])[features]

# One-hot encode & align columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)

# === 5. Scale and shape for LSTM ===
X_scaled = feat_scaler.transform(input_encoded)
X_seq = np.zeros((1, 30, X_scaled.shape[1]))
X_seq[0, -1, :] = X_scaled[0]

# === 6. Predict ===
y_pred_scaled = model.predict(X_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled)

print(f"\n💰 Predicted Price per m²: {y_pred[0][0]:,.2f} TND")
