import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd

# ---------------- CONFIG ----------------
SEQ_LEN = 20
TRAINED_FUTURE_STEPS = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 3
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model/lstm_forecast_multistep.pth"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/features.pkl"
DATA_PATH = "data/cleaned_metrics1.csv"  # CSV containing historical metrics

# ---------------- LOAD SCALER & FEATURES ----------------
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
INPUT_SIZE = len(FEATURES)

# ---------------- MODEL ----------------
class LSTMForecastMulti(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3,
                 future_steps=TRAINED_FUTURE_STEPS, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size * future_steps)
        self.input_size = input_size
        self.future_steps = future_steps

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.future_steps, self.input_size)

# ---------------- LOAD EXISTING MODEL ----------------
lstm_model = LSTMForecastMulti(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, TRAINED_FUTURE_STEPS).to(DEVICE)
lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
lstm_model.eval()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
df_features = df[FEATURES]

# Only retrain if enough rows
if len(df_features) < SEQ_LEN:
    print(f"Not enough rows to retrain. Need at least {SEQ_LEN}.")
    exit()

# ---------------- CHECK IF RETRAIN REQUIRED ----------------
RETRAIN_INTERVAL = 10
if len(df_features) % RETRAIN_INTERVAL != 0:
    print(f"Not retraining. Current rows ({len(df_features)}) not a multiple of {RETRAIN_INTERVAL}.")
    exit()

# ---------------- PREPARE TRAINING SEQUENCES ----------------
X_scaled = scaler.transform(df_features.values)
X_seq, y_seq = [], []
for i in range(len(X_scaled) - SEQ_LEN):
    X_seq.append(X_scaled[i:i+SEQ_LEN])
    y_seq.append(X_scaled[i+SEQ_LEN])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(DEVICE)

# ---------------- RETRAIN MODEL ----------------
model_retrain = LSTMForecastMulti(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, TRAINED_FUTURE_STEPS).to(DEVICE)
optimizer = torch.optim.Adam(model_retrain.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model_retrain.train()
    optimizer.zero_grad()
    output = model_retrain(X_tensor)
    loss = criterion(output.view(-1, INPUT_SIZE), y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item():.4f}")

# ---------------- SAVE UPDATED MODEL ----------------
lstm_model.load_state_dict(model_retrain.state_dict())
torch.save(lstm_model.state_dict(), MODEL_PATH)
print("✅ LSTM model retrained and saved successfully!")
