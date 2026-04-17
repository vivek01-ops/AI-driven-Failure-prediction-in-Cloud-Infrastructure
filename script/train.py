import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ---------------- CONFIG ----------------
SEQ_LEN = 20
FUTURE_STEPS = 10
EPOCHS = 100
LR = 0.001
PATIENCE = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_forecast_multistep.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/30000_data.csv", parse_dates=["timestamp"])
df = df.drop(columns=["timestamp"])  # only numeric features

# drop constant / static metrics
drop_cols = ["Memory_GiB_Total", "Disk_GiB_Total"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

FEATURES = df.columns.tolist()
n_features = len(FEATURES)

# ---------------- SCALE ----------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.values)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(FEATURES, FEATURES_PATH)

# ---------------- PREPARE SEQUENCES ----------------
X, y = [], []
for i in range(len(data_scaled) - SEQ_LEN - FUTURE_STEPS + 1):
    X.append(data_scaled[i:i+SEQ_LEN])
    y.append(data_scaled[i+SEQ_LEN:i+SEQ_LEN+FUTURE_STEPS])
X, y = np.array(X), np.array(y)

print("X shape:", X.shape)  # (samples, 20, features)
print("y shape:", y.shape)  # (samples, 10, features)

# split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class LSTMForecastMulti(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, future_steps=FUTURE_STEPS, dropout=0.2):
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

model = LSTMForecastMulti(input_size=n_features).to(DEVICE)

# ---------------- TRAINING SETUP ----------------
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

best_val_loss = float("inf")
patience_counter = 0
best_preds, best_true = None, None

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    # training
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # validation
    model.eval()
    val_losses = []
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_output = model(xb)
            val_loss = criterion(val_output, yb)
            val_losses.append(val_loss.item())
            all_preds.append(val_output.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
        best_preds, best_true = all_preds, all_true
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("✅ Best model saved with val_loss:", best_val_loss)

# ---------------- FINAL EVALUATION ----------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds_flat = np.vstack([p.reshape(-1, n_features) for p in best_preds])
all_true_flat = np.vstack([t.reshape(-1, n_features) for t in best_true])

all_preds_inv = scaler.inverse_transform(all_preds_flat)
all_true_inv = scaler.inverse_transform(all_true_flat)

print("\n📊 Validation Metrics (per feature):")
for i, col in enumerate(FEATURES):
    r2 = r2_score(all_true_inv[:, i], all_preds_inv[:, i])
    mae = mean_absolute_error(all_true_inv[:, i], all_preds_inv[:, i])
    rmse = np.sqrt(mean_squared_error(all_true_inv[:, i], all_preds_inv[:, i]))
    print(f"{col:25} | R²: {r2:8.3f} | MAE: {mae:7.3f} | RMSE: {rmse:7.3f}")

# ---------------- FINAL EVALUATION ----------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds_flat = np.vstack([p.reshape(-1, n_features) for p in best_preds])
all_true_flat = np.vstack([t.reshape(-1, n_features) for t in best_true])

all_preds_inv = scaler.inverse_transform(all_preds_flat)
all_true_inv = scaler.inverse_transform(all_true_flat)

print("\n📊 Validation Metrics (per feature):")

metrics_data = []

for i, col in enumerate(FEATURES):
    r2 = r2_score(all_true_inv[:, i], all_preds_inv[:, i])
    mae = mean_absolute_error(all_true_inv[:, i], all_preds_inv[:, i])
    rmse = np.sqrt(mean_squared_error(all_true_inv[:, i], all_preds_inv[:, i]))

    print(f"{col:25} | R²: {r2:8.3f} | MAE: {mae:7.3f} | RMSE: {rmse:7.3f}")

    # ✅ store for plots
    metrics_data.append([col, r2, mae, rmse])

# ✅ FIX: create dataframe
metrics_df = pd.DataFrame(metrics_data, columns=["Feature", "R2", "MAE", "RMSE"])


# ================= CLEAN BENCHMARKS (MAX 7) =================
from sklearn.metrics import roc_curve, auc

BENCHMARK_DIR = "benchmarks"
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ---------------- 1. LOSS CURVE ----------------
# NOTE: uses last epoch batch losses (since training loop unchanged)
plt.figure()
plt.plot(train_losses, label="Train Loss (last epoch)")
plt.plot(val_losses, label="Val Loss (last epoch)")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig(f"{BENCHMARK_DIR}/loss_curve.png")
plt.close()

# ---------------- 2. R2 ----------------
plt.figure()
metrics_df.set_index("Feature")[["R2"]].plot(kind="bar")
plt.title("R2 Score per Feature")
plt.tight_layout()
plt.savefig(f"{BENCHMARK_DIR}/r2.png")
plt.close()

# ---------------- 3. RMSE ----------------
plt.figure()
metrics_df.set_index("Feature")[["RMSE"]].plot(kind="bar")
plt.title("RMSE per Feature")
plt.tight_layout()
plt.savefig(f"{BENCHMARK_DIR}/rmse.png")
plt.close()

# ---------------- 4. PRED VS ACTUAL ----------------
important = ["CPU_percent", "Memory_Used_Percent", "Disk_Used_Percent"]

for col in important:
    if col not in FEATURES:
        continue

    i = FEATURES.index(col)

    plt.figure(figsize=(10,4))
    plt.plot(all_true_inv[:300, i], label="Actual")
    plt.plot(all_preds_inv[:300, i], label="Predicted")
    plt.title(f"{col} - Prediction vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BENCHMARK_DIR}/pred_vs_actual_{col}.png")
    plt.close()

# ---------------- 5. RESIDUAL ----------------
errors = all_preds_inv - all_true_inv

plt.figure()
plt.scatter(all_true_inv.flatten(), errors.flatten(), alpha=0.2)
plt.axhline(0)
plt.title("Residual Plot")
plt.xlabel("Actual")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig(f"{BENCHMARK_DIR}/residual.png")
plt.close()

# ---------------- 6. ROC-AUC ----------------
THRESHOLDS = {
    "CPU_percent": 80,
    "Memory_Used_Percent": 75,
    "Disk_Used_Percent": 80
}

for col in THRESHOLDS:
    if col not in FEATURES:
        continue

    i = FEATURES.index(col)

    y_true = (all_true_inv[:, i] > THRESHOLDS[col]).astype(int)
    y_score = all_preds_inv[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BENCHMARK_DIR}/roc_{col}.png")
    plt.close()

# ---------------- 7. CONFUSION MATRIX ----------------
for col in THRESHOLDS:
    if col not in FEATURES:
        continue

    i = FEATURES.index(col)

    y_true = (all_true_inv[:, i] > THRESHOLDS[col]).astype(int)
    y_pred = (all_preds_inv[:, i] > THRESHOLDS[col]).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {col}")
    plt.tight_layout()
    plt.savefig(f"{BENCHMARK_DIR}/cm_{col}.png")
    plt.close()

print("✅ Clean research benchmarks saved")