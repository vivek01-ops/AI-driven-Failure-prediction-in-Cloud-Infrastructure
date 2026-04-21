import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ======================================================
# CONFIG
# ======================================================
PROMETHEUS_BASE = "http://192.168.49.2:30477/api/v1"
PROM_URL = f"{PROMETHEUS_BASE}/query"

SCRAPE_INTERVAL = 5
SEQ_LEN = 40
FUTURE_STEPS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"

# ======================================================
# STREAMLIT
# ======================================================
st.set_page_config(layout="wide")
st.title("Infrastructure Monitoring with LSTM Models")

# ======================================================
# MODEL CLASS (same as trainer)
# ======================================================
class LSTMForecast(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(128, FUTURE_STEPS)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ======================================================
# LOAD MODELS
# ======================================================
TARGETS = ["cpu", "memory", "disk", "node1", "node5", "node15"]

MODELS = {}
SCALERS = {}

for t in TARGETS:
    try:
        model = LSTMForecast().to(DEVICE)
        model.load_state_dict(
            torch.load(f"{MODEL_DIR}/{t}_model.pth", map_location=DEVICE)
        )
        model.eval()

        scaler = joblib.load(f"{MODEL_DIR}/{t}_scaler.pkl")

        MODELS[t] = model
        SCALERS[t] = scaler

    except:
        pass


# ======================================================
# PROMETHEUS QUERIES
# ======================================================
METRICS = {
    "cpu": '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',

    "mem_avail": "node_memory_MemAvailable_bytes",
    "mem_total": "node_memory_MemTotal_bytes",

    "disk_avail": 'avg(node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"})',
    "disk_total": 'avg(node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})',

    "node1": "avg(node_load1)",
    "node5": "avg(node_load5)",
    "node15": "avg(node_load15)",
}


# ======================================================
# HELPERS
# ======================================================
def query_prom(query):
    try:
        r = requests.get(PROM_URL, params={"query": query}, timeout=3)
        data = r.json()

        if data["status"] == "success":
            vals = data["data"]["result"]

            if vals:
                nums = [float(v["value"][1]) for v in vals]
                return np.mean(nums)

    except:
        return None

    return None


def scrape_metrics():

    row = {"timestamp": datetime.now()}

    cpu = query_prom(METRICS["cpu"])
    mem_avail = query_prom(METRICS["mem_avail"])
    mem_total = query_prom(METRICS["mem_total"])

    disk_avail = query_prom(METRICS["disk_avail"])
    disk_total = query_prom(METRICS["disk_total"])

    node1 = query_prom(METRICS["node1"])
    node5 = query_prom(METRICS["node5"])
    node15 = query_prom(METRICS["node15"])

    if cpu is not None:
        row["CPU_percent"] = cpu

    if mem_avail and mem_total:
        used = mem_total - mem_avail
        row["Memory_GiB_Used"] = used / (1024**3)
        row["Memory_Used_Percent"] = used / mem_total * 100

    if disk_avail and disk_total:
        used = disk_total - disk_avail
        row["Disk_GiB_Used"] = used / (1024**3)
        row["Disk_Used_Percent"] = used / disk_total * 100

    if node1 is not None:
        row["node1"] = node1

    if node5 is not None:
        row["node5"] = node5

    if node15 is not None:
        row["node15"] = node15

    return row


# ======================================================
# FORECAST ONE METRIC
# ======================================================
def predict_metric(series, metric_name):

    if len(series) < SEQ_LEN:
        return []

    arr = np.array(series[-SEQ_LEN:]).reshape(-1, 1)

    scaler = SCALERS[metric_name]
    model = MODELS[metric_name]

    scaled = scaler.transform(arr)

    x = torch.tensor(
        scaled[np.newaxis, :, :],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    return pred


# ======================================================
# CHART FUNCTION
# ======================================================
# ======================================================
# REPLACE OLD show_chart() FUNCTION WITH THIS
# ======================================================

def show_chart(df, col, metric_key, title):

    st.subheader(title, divider="red")

    actual = df[["timestamp", col]].copy()
    pred = predict_metric(df[col].dropna().tolist(), metric_key)

    current_actual = df[col].iloc[-1]

    if len(pred) > 0:

        # ----------------------------
        # current prediction (next step)
        # ----------------------------
        current_pred = pred[0]

        # ----------------------------
        # future selectable point
        # ----------------------------
        future_option = st.selectbox(
            f"{title} Future Time",
            ["1 min", "2 min"],
            key=f"{metric_key}_future"
        )

        mins = int(future_option.split()[0])

        # if refresh = 5 sec
        # 1 min = 12 steps
        # 2 min = 24 steps
        step_index = int((mins * 60) / SCRAPE_INTERVAL) - 1

        # if model only predicts 10 steps, cap last step
        step_index = min(step_index, len(pred) - 1)

        future_pred = pred[step_index]

        # ----------------------------
        # METRICS
        # ----------------------------
        delta1_val = current_pred - current_actual
        delta2_val = future_pred - current_actual
        m1, m2, m3 = st.columns(3, border=True)
        is_percent = metric_key in ["cpu", "memory", "disk"]

        suffix = "%" if is_percent else ""


        with m1:
            st.metric(
                "Current Actual",
                f"{current_actual:.2f}{suffix}"
            )

        with m2:
            st.metric(
                "Current Predicted",
                f"{current_pred:.2f}{suffix}",
                delta=f"{delta1_val:+.2f}{suffix} vs actual"
            )

        with m3:
            st.metric(
                f"Predicted After {future_option}",
                f"{future_pred:.2f}{suffix}",
                delta=f"{delta2_val:+.2f}{suffix} vs actual"
            )

        # ----------------------------
        # CHART
        # ----------------------------
        last_time = df["timestamp"].iloc[-1]

        future_times = [
            last_time + timedelta(seconds=SCRAPE_INTERVAL*(i+1))
            for i in range(FUTURE_STEPS)
        ]

        is_percent = metric_key in ["cpu", "memory", "disk"]

        actual_name = "Actual %" if is_percent else "Actual"
        pred_name = "Predicted %" if is_percent else "Predicted"

        pred_df = pd.DataFrame({
            "timestamp": future_times,
            pred_name: pred
        })

        actual = actual.rename(columns={col: actual_name})
        actual = actual.set_index("timestamp")
        pred_df = pred_df.set_index("timestamp")

        merged = actual.merge(
            pred_df,
            left_index=True,
            right_index=True,
            how="outer"
        )

        st.area_chart(merged, width="stretch")

    else:
        st.metric("Current Actual", f"{current_actual:.2f}", border=True)
        st.line_chart(actual.set_index("timestamp"), width="stretch")


# ======================================================
# SIDEBAR
# ======================================================
col1, col2 = st.columns(2)

with col1:
    interval = st.number_input(
        "Refresh Seconds",
        min_value=1,
        value=SCRAPE_INTERVAL
    )

with col2:
    rows = st.selectbox(
        "Rows",
        [30, 50, 100, 200],
        index=1
    )

st_autorefresh(interval=interval*1000, key="refresh")


# ======================================================
# SESSION DATAFRAME
# ======================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

new_row = scrape_metrics()

st.session_state.df = pd.concat(
    [st.session_state.df, pd.DataFrame([new_row])],
    ignore_index=True
)

df = st.session_state.df.tail(rows).copy()


# ======================================================
# METRICS
# ======================================================
# c1, c2, c3 = st.columns(3)

# with c1:
#     if "CPU_percent" in df.columns:
#         st.metric("CPU %", f"{df['CPU_percent'].iloc[-1]:.2f}")

# with c2:
#     if "Memory_Used_Percent" in df.columns:
#         st.metric("Memory %", f"{df['Memory_Used_Percent'].iloc[-1]:.2f}")

# with c3:
#     if "Disk_Used_Percent" in df.columns:
#         st.metric("Disk %", f"{df['Disk_Used_Percent'].iloc[-1]:.2f}")


# ======================================================
# MAIN CHARTS
# ======================================================

a, b, c = st.columns(3, border=True)

with a:
    if "Memory_Used_Percent" in df.columns:
        show_chart(
            df,
            "Memory_Used_Percent",
            "memory",
            "Memory Usage (%)"
        )

with b:
    if "CPU_percent" in df.columns:
        show_chart(
            df,
            "CPU_percent",
            "cpu",
            "CPU Usage (%)"
        )

with c:
    if "Disk_Used_Percent" in df.columns:
        show_chart(
            df,
            "Disk_Used_Percent",
            "disk",
            "Disk Usage (%)"
        )


# ======================================================
# NODE LOADS
# ======================================================
st.header("Node Loads")

n1, n2, n3 = st.columns(3, border=True)

with n1:
    if "node1" in df.columns:
        show_chart(df, "node1", "node1", "Load 1")

with n2:
    if "node5" in df.columns:
        show_chart(df, "node5", "node5", "Load 5")

with n3:
    if "node15" in df.columns:
        show_chart(df, "node15", "node15", "Load 15")


# ======================================================
# TABLE
# ======================================================
st.subheader("Recent Data", divider="red")
st.dataframe(df.tail(20), width="stretch")