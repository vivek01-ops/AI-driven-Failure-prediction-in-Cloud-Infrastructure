import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# ==================================================
# CONFIG
# ==================================================
SEQ_LEN = 40
FUTURE_STEPS = 10
EPOCHS = 100
LR = 0.001
PATIENCE = 10
BATCH_SIZE = 64
HIDDEN = 128
LAYERS = 3
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/30000_data.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# remove timestamp
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# ==================================================
# AUTO DETECT REQUIRED FEATURES
# ==================================================
def find_col(keyword):
    keyword = keyword.lower()
    for c in df.columns:
        if keyword in c.lower():
            return c
    return None


TARGETS = {
    "cpu": find_col("cpu_percent"),
    "memory": find_col("memory_used_percent"),
    "disk": find_col("disk_used_percent"),
    "node1": find_col("load1"),
    "node5": find_col("load5"),
    "node15": find_col("load15"),
}

# remove missing
TARGETS = {k: v for k, v in TARGETS.items() if v is not None}

print("\nDetected Metrics:")
for k, v in TARGETS.items():
    print(f"{k:10} --> {v}")

# ==================================================
# MODEL
# ==================================================
class LSTMForecast(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.fc = nn.Linear(HIDDEN, FUTURE_STEPS)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ==================================================
# CREATE SEQUENCES
# ==================================================
def make_sequences(arr):
    X, y = [], []

    for i in range(len(arr) - SEQ_LEN - FUTURE_STEPS + 1):
        X.append(arr[i:i+SEQ_LEN])
        y.append(arr[i+SEQ_LEN:i+SEQ_LEN+FUTURE_STEPS])

    return np.array(X), np.array(y)


# ==================================================
# TRAIN ONE MODEL
# ==================================================
def train_metric(metric_name, column_name):

    print("\n" + "="*60)
    print(f"Training {metric_name.upper()} --> {column_name}")
    print("="*60)

    series = df[column_name].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(series)

    X, y = make_sequences(scaled)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train.squeeze(-1), dtype=torch.float32).to(DEVICE)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val.squeeze(-1), dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = LSTMForecast().to(DEVICE)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )

    best_loss = float("inf")
    patience_counter = 0
    best_preds = []
    best_true = []

    model_path = f"{MODEL_DIR}/{metric_name}_model.pth"
    scaler_path = f"{MODEL_DIR}/{metric_name}_scaler.pkl"

    # ------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------
    for epoch in range(EPOCHS):

        model.train()
        train_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        preds = []
        truths = []

        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)

                loss = criterion(out, yb)
                val_losses.append(loss.item())

                preds.append(out.cpu().numpy())
                truths.append(yb.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), model_path)
            best_preds = preds
            best_true = truths

        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    # ------------------------------------------
    # SAVE SCALER
    # ------------------------------------------
    joblib.dump(scaler, scaler_path)

    # ------------------------------------------
    # METRICS
    # ------------------------------------------
    pred = np.vstack(best_preds)
    true = np.vstack(best_true)

    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(true.reshape(-1, 1)).flatten()

    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    print(f"\n{metric_name.upper()} RESULTS")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    print(f"Saved: {model_path}")
    print(f"Saved: {scaler_path}")


# ==================================================
# TRAIN ALL METRICS
# ==================================================
for metric, column in TARGETS.items():
    train_metric(metric, column)

print("\nAll separate models trained successfully.")