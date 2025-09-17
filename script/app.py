import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import subprocess
import joblib
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
PROMETHEUS_BASE = "http://192.168.49.2:32600/api/v1"
PROMETHEUS_URL = f"{PROMETHEUS_BASE}/query"
SCRAPE_INTERVAL = 1  # seconds
SEQ_LEN = 20
TRAINED_FUTURE_STEPS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model/lstm_forecast_multistep.pth"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/features.pkl"

# ---------------- LOAD SCALER & FEATURES ----------------
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

# Drop static / capacity features (not used in training)
DROP_FEATURES = ["Memory_GiB_Total", "Disk_GiB_Total"]
FEATURES = [f for f in FEATURES if f not in DROP_FEATURES]

INPUT_SIZE = len(FEATURES)


# ---------------- METRICS ----------------
METRICS_RAW = {
    "CPU_percent": '100 - (avg by (instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
    "node_memory_MemAvailable_bytes": "node_memory_MemAvailable_bytes",
    "node_memory_MemTotal_bytes": "node_memory_MemTotal_bytes",
    "node_filesystem_size_bytes": 'avg(node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})',
    "node_filesystem_avail_bytes": 'avg(node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"})'
}
NODE_LOAD_METRICS = ["node_load1", "node_load5", "node_load15"]

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

# Init + Load
lstm_model = LSTMForecastMulti(INPUT_SIZE, 128, 3, TRAINED_FUTURE_STEPS, 0.2).to(DEVICE)
lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
lstm_model.eval()

# ---------------- HELPERS ----------------
def query_prometheus(query):
    try:
        r = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=3)
        result = r.json()
        if result.get("status") == "success":
            values = result["data"]["result"]
            if values:
                vals = [float(v["value"][1]) for v in values]
                return sum(vals) / len(vals)
        return None
    except Exception:
        return None

def get_node_loads():
    loads = {}
    for metric in NODE_LOAD_METRICS:
        try:
            r = requests.get(PROMETHEUS_URL, params={"query": metric}, timeout=3)
            result = r.json()
            if result.get("status") == "success":
                for item in result["data"]["result"]:
                    node = item["metric"].get("instance", "unknown")
                    value = float(item["value"][1])
                    loads[f"{node}_{metric}"] = value
        except Exception:
            continue
    return loads

def scrape_metrics():
    data = {"timestamp": datetime.now()}
    for name, query in METRICS_RAW.items():
        value = query_prometheus(query)
        if value is not None:
            data[name] = value
    data.update(get_node_loads())
    return data

def process_metrics(df):
    if df.empty:
        return df
    if "node_memory_MemAvailable_bytes" in df.columns and "node_memory_MemTotal_bytes" in df.columns:
        df["Memory_GiB_Available"] = df["node_memory_MemAvailable_bytes"] / (1024**3)
        df["Memory_GiB_Total"] = df["node_memory_MemTotal_bytes"] / (1024**3)
        df["Memory_GiB_Used"] = df["Memory_GiB_Total"] - df["Memory_GiB_Available"]
        df["Memory_Used_Percent"] = (df["Memory_GiB_Used"] / df["Memory_GiB_Total"]) * 100
    if "node_filesystem_size_bytes" in df.columns and "node_filesystem_avail_bytes" in df.columns:
        df["Disk_GiB_Total"] = df["node_filesystem_size_bytes"] / (1024**3)
        df["Disk_GiB_Available"] = df["node_filesystem_avail_bytes"] / (1024**3)
        df["Disk_GiB_Used"] = df["Disk_GiB_Total"] - df["Disk_GiB_Available"]
        df["Disk_Used_Percent"] = (df["Disk_GiB_Used"] / df["Disk_GiB_Total"]) * 100
    return df

def make_forecast(df_display, interval):
    """
    Forecast metrics starting at t=0 up to current time + future horizon.
    Ensures predicted curve is longer than actual.
    """
    if len(df_display) < SEQ_LEN:
        return pd.DataFrame()

    missing = [f for f in FEATURES if f not in df_display.columns]
    if missing:
        st.warning(f"Skipping forecast — missing features in live data: {missing}")
        return pd.DataFrame()

    try:
        features = df_display[FEATURES].values
        timestamps = df_display["timestamp"].values

        preds_aligned = []
        times_aligned = []

        # Step 1: rolling predictions aligned with actuals
        for i in range(len(features) - SEQ_LEN):
            seq = features[i:i+SEQ_LEN]
            seq_scaled = scaler.transform(seq)
            X = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                pred_scaled = lstm_model(X).cpu().numpy()[0]

            pred = scaler.inverse_transform(pred_scaled)

            # Only keep the first-step prediction to align with actual
            preds_aligned.append(pred[0])
            times_aligned.append(timestamps[i+SEQ_LEN])  # align with next time step

        # Step 2: future horizon prediction from the last window
        last_seq = features[-SEQ_LEN:]
        seq_scaled = scaler.transform(last_seq)
        X = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_scaled = lstm_model(X).cpu().numpy()[0]

        future_pred = scaler.inverse_transform(pred_scaled)

        last_time = df_display["timestamp"].iloc[-1]
        future_times = [last_time + timedelta(seconds=interval*(i+1)) for i in range(TRAINED_FUTURE_STEPS)]

        preds_full = np.vstack([preds_aligned, future_pred])
        times_full = list(times_aligned) + future_times

        forecast_df = pd.DataFrame(preds_full, columns=FEATURES)
        forecast_df["timestamp"] = times_full

        return forecast_df

    except Exception as e:
        st.warning(f"LSTM forecast error: {e}")
        return pd.DataFrame()

def overlay_chart(actual_df, pred_df, y_col, title, chart_type="line", time_window=None):
    """
    Overlay actual and predicted values within a specific time window.
    """
    st.subheader(title, divider="red")

    # Apply time window to actual data
    df_actual = actual_df.copy()
    if time_window is not None:
        df_actual = df_actual[df_actual["timestamp"] >= datetime.now() - time_window]

    df_plot = pd.DataFrame({
        "timestamp": df_actual["timestamp"],
        "Actual": df_actual[y_col]
    }).set_index("timestamp")

    # Apply same time window to predictions
    if not pred_df.empty and y_col in pred_df.columns:
        df_pred = pred_df.copy()
        if time_window is not None:
            df_pred = df_pred[df_pred["timestamp"] >= datetime.now() - time_window]

        pred_series = pd.Series(
            df_pred[y_col].values,
            index=df_pred["timestamp"]
        )

        # Merge actual and predicted
        df_plot = df_plot.merge(pred_series.rename("Predicted"), left_index=True, right_index=True, how="outer")

    # Chart rendering
    if chart_type == "area":
        st.line_chart(df_plot, use_container_width=True)
    else:
        st.line_chart(df_plot, use_container_width=True)
   
# ---------------- STREAMLIT APP ----------------
st.set_page_config(layout="wide")
st.title("Real-Time Prometheus Dashboard with LSTM Forecasts")

# Sidebar
col1, col2 = st.columns(2, gap="medium")
with col1:
    interval = st.number_input("Scrape Interval (sec)", min_value=1, value=SCRAPE_INTERVAL)
with col2:
    time_window_option = st.selectbox(
        "Time Window",
        ("30 sec","1 min","5 min","10 min","30 min","1 hr","2 hr","5 hr","6 hr","8 hr","12 hr","18 hr","24 hr")
    )

num, unit = time_window_option.split()
num = int(num)
if unit.startswith("sec"): time_delta = timedelta(seconds=num)
elif unit.startswith("min"): time_delta = timedelta(minutes=num)
else: time_delta = timedelta(hours=num)

st_autorefresh(interval=interval * 1000, key="auto_refresh")

# Data init
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["timestamp"] + list(METRICS_RAW.keys()) + NODE_LOAD_METRICS)

new_row = scrape_metrics()
if not new_row:
    st.warning("No metrics scraped yet — waiting for Prometheus.")
    st.stop()
if st.session_state.df.empty or new_row != st.session_state.df.iloc[-1].to_dict():
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)

df_display = process_metrics(st.session_state.df.copy())
df_recent = df_display[df_display["timestamp"] >= datetime.now()-time_delta]

future_preds_df = make_forecast(df_display, interval)

# ---------------- PREDICTION CONFIDENCE METRIC ----------------
colp, colm = st.columns(2)

# Initialize histories if not exists
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []

# ---------------- REAL-TIME PREDICTION CONFIDENCE ----------------
available_features = [f for f in FEATURES if f in df_recent.columns]
pred_available_features = [f for f in FEATURES if (not future_preds_df.empty and f in future_preds_df.columns)]

# Intersection of features present in both actual recent and prediction frames
common_features = [f for f in available_features if f in pred_available_features]

if len(df_recent) >= 1 and not future_preds_df.empty and common_features:
    last_actual = df_recent[common_features].iloc[-1].values
    last_pred = future_preds_df[common_features].iloc[0].values

    # Real-time per-metric confidence
    confidences = [max(0, 100 - (abs(pred - actual)/max(abs(actual), 1))*100)
                   for actual, pred in zip(last_actual, last_pred)]
    prediction_confidence = np.mean(confidences) if confidences else 0.0

    # Update confidence history
    st.session_state.confidence_history.append(prediction_confidence)
    if len(st.session_state.confidence_history) > 50:
        st.session_state.confidence_history = st.session_state.confidence_history[-50:]

    # Delta for prediction confidence
    confidence_delta = (st.session_state.confidence_history[-1] -
                        st.session_state.confidence_history[-2]) if len(st.session_state.confidence_history) > 1 else 0

    with colp:
        st.metric(
            label="Prediction Confidence (%)",
            value=f"{prediction_confidence:.2f}%",
            delta=f"{confidence_delta:.2f}%",
            chart_data=st.session_state.confidence_history,
            chart_type="line",
            help="Shows real-time prediction confidence",
            border=True,
        )
else:
    with colp:
        st.metric(
            label="Prediction Confidence (%)",
            value="N/A",
            delta="N/A",
            help="Shows real-time prediction confidence",
            border=True
        )

# ---------------- MODEL ACCURACY (updated only on retrain) ----------------
# Use last stored retrain accuracy if exists
if "last_model_accuracy" not in st.session_state:
    st.session_state.last_model_accuracy = None

if st.session_state.last_model_accuracy is not None:
    model_accuracy = st.session_state.last_model_accuracy
    st.session_state.accuracy_history.append(model_accuracy)
    if len(st.session_state.accuracy_history) > 50:
        st.session_state.accuracy_history = st.session_state.accuracy_history[-50:]

    # Delta for model accuracy
    accuracy_delta = (st.session_state.accuracy_history[-1] -
                      st.session_state.accuracy_history[-2]) if len(st.session_state.accuracy_history) > 1 else 0

    with colm:
        st.metric(
            label="Model Accuracy (%)",
            value=f"{model_accuracy:.2f}%",
            delta=f"{accuracy_delta:.2f}%",
            chart_data=st.session_state.accuracy_history,
            chart_type="line",
            help="Model accuracy after last retraining",
            border=True
        )
else:
    with colm:
        st.metric(
            label="Model Accuracy (%)",
            value="N/A",
            delta="N/A",
            help="Model accuracy after last retraining",
            border=True
        )

# Charts
cola, colb, colc = st.columns(3, border=True)

with cola:
    if "Memory_GiB_Used" in df_recent.columns:
        overlay_chart(df_display, future_preds_df, "Memory_GiB_Used", 
                      f"Memory Usage (GiB) ({time_window_option})", 
                      chart_type="area", time_window=time_delta)
with colb:
    if "CPU_percent" in df_recent.columns:
        overlay_chart(df_display, future_preds_df, "CPU_percent", 
                      f"CPU Usage (%) ({time_window_option})", 
                      chart_type="line", time_window=time_delta)
with colc:
    if "Disk_GiB_Used" in df_recent.columns:
        overlay_chart(df_display, future_preds_df, "Disk_GiB_Used", 
                      f"Disk Usage (GiB) ({time_window_option})", 
                      chart_type="area", time_window=time_delta)

# ---- NODE LOAD METRICS (3 charts per row per node) ----
# ---- NODE LOAD METRICS (aligned per-column with time window) ----
load_cols = [c for c in df_recent.columns if "_node_load" in c]

if load_cols:
    nodes = sorted({c.split("_node_load")[0] for c in load_cols})
    st.header(f"Node Load Averages ({time_window_option})")

    # Ensure predictions timestamps are datetime if preds exist
    if not future_preds_df.empty and "timestamp" in future_preds_df.columns:
        future_preds_df["timestamp"] = pd.to_datetime(future_preds_df["timestamp"])

    for node in nodes:
        st.subheader(f"Node: `{node}`")
        col1, col2, col3 = st.columns(3, border=True)

        for load_metric, col in zip(["load1", "load5", "load15"], [col1, col2, col3]):
            colname = f"{node}_node_{load_metric}"
            if colname not in df_recent.columns:
                continue

            with col:
                st.subheader(f"**{load_metric.upper()}**", divider="red")

                # Filter actual and predicted by time window
                df_actual = df_display.copy()
                df_actual = df_actual[df_actual["timestamp"] >= datetime.now() - time_delta]

                df_plot = pd.DataFrame({
                    "timestamp": df_actual["timestamp"],
                    "Actual": df_actual[colname]
                }).set_index("timestamp")

                # Predictions filtered by time window
                if not future_preds_df.empty and colname in future_preds_df.columns:
                    df_pred = future_preds_df.copy()
                    df_pred = df_pred[df_pred["timestamp"] >= datetime.now() - time_delta]

                    pred_series = pd.Series(
                        df_pred[colname].values,
                        index=pd.to_datetime(df_pred["timestamp"])
                    )

                    # Merge actual and predicted
                    df_plot = df_plot.merge(pred_series.rename("Predicted"),
                                            left_index=True, right_index=True, how="outer")

                # Cast to float for safety
                df_plot = df_plot.astype(float)

                # Show chart
                st.line_chart(df_plot, use_container_width=True)



# Data Table
st.subheader("Recent Data", divider="red")
rown = st.selectbox("Show last N rows", (10,20,30,40,50,100))
useful_cols = [c for c in df_display.columns if c in [
    "timestamp", "CPU_percent",
    "Memory_GiB_Used","Memory_Used_Percent",
    "Disk_GiB_Used","Disk_Used_Percent"
] or "_node_load" in c]
st.dataframe(df_display[useful_cols].tail(rown))

# ---------------- LSTM RETRAIN AFTER EVERY RETRAIN_INTERVAL ENTRIES ----------------
RETRAIN_INTERVAL = 10  # retrain after every 10 new rows

# Initialize session state variables
if "last_retrain_row_count" not in st.session_state:
    st.session_state.last_retrain_row_count = 0
if "retrain_in_progress" not in st.session_state:
    st.session_state.retrain_in_progress = False

# Calculate new rows since last retrain
num_new_rows = len(st.session_state.df) - st.session_state.last_retrain_row_count

# Trigger retraining
if num_new_rows >= RETRAIN_INTERVAL and not st.session_state.retrain_in_progress:
    st.session_state.retrain_in_progress = True
    st.toast("🔄 Retraining LSTM model with latest data...")

    try:
        # Call external retrain.py
        subprocess.run(["python", "retrain.py"], check=True)
        st.toast("✅ LSTM retrained!")
    except subprocess.CalledProcessError as e:
        st.toast(f"Retraining failed: {e}")

    # Update the row count tracker
    st.session_state.last_retrain_row_count = len(st.session_state.df)
    st.session_state.retrain_in_progress = False
