import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
SEQ_LEN = 40
FUTURE_STEPS = 10
SCRAPE_INTERVAL = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"
PROM_URL = "http://192.168.49.2:30477/api/v1/query"

# =========================
# PAGE
# =========================
st.set_page_config(layout="wide")
st.title("📊 AI Infrastructure Monitoring Dashboard")

# =========================
# MODEL
# =========================
class LSTMForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 3, batch_first=True)
        self.fc = nn.Linear(128, FUTURE_STEPS)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================
# LOAD MODELS
# =========================
TARGETS = ["cpu","memory","disk","node1","node5","node15"]
MODELS, SCALERS = {}, {}

for t in TARGETS:
    try:
        m = LSTMForecast().to(DEVICE)
        m.load_state_dict(torch.load(f"{MODEL_DIR}/{t}_model.pth", map_location=DEVICE))
        m.eval()
        MODELS[t] = m
        SCALERS[t] = joblib.load(f"{MODEL_DIR}/{t}_scaler.pkl")
    except Exception as e:
        st.warning(f"{t} model not loaded")

# =========================
# PROMETHEUS
# =========================
def query(q):
    try:
        r = requests.get(PROM_URL, params={"query": q}, timeout=3)
        data = r.json()["data"]["result"]
        if data:
            return np.mean([float(x["value"][1]) for x in data])
    except:
        return None
    return None

def check_prometheus():
    try:
        r = requests.get(
            PROM_URL,
            params={"query": "up"},   # ✅ VALID QUERY
            timeout=2
        )

        data = r.json()

        # Prometheus success condition
        return (
            r.status_code == 200 and
            data.get("status") == "success"
        )

    except:
        return False

# =========================
# SCRAPE
# =========================
def scrape():
    row = {"timestamp": datetime.now()}

    cpu = query('100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    mem_a = query("node_memory_MemAvailable_bytes")
    mem_t = query("node_memory_MemTotal_bytes")
    disk_a = query("node_filesystem_avail_bytes")
    disk_t = query("node_filesystem_size_bytes")

    row["CPU_percent"] = cpu
    row["Memory_Used_Percent"] = ((mem_t - mem_a)/mem_t)*100 if mem_a and mem_t else None
    row["Disk_Used_Percent"] = ((disk_t - disk_a)/disk_t)*100 if disk_a and disk_t else None

    row["node1"] = query("node_load1")
    row["node5"] = query("node_load5")
    row["node15"] = query("node_load15")

    return row

# =========================
# PREDICT (STRICT 40)
# =========================
def predict(series, key):
    if key not in MODELS:
        return []

    series = pd.Series(series).dropna()

    if len(series) < SEQ_LEN:
        return []

    arr = np.array(series[-SEQ_LEN:]).reshape(-1, 1)

    scaler = SCALERS[key]
    model = MODELS[key]

    x = scaler.transform(arr)
    x = torch.tensor(x[np.newaxis,:,:], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    return scaler.inverse_transform(pred.reshape(-1,1)).flatten()

# =========================
# HEALTH
# =========================
def calculate_health(df):
    vals = []

    if "CPU_percent" in df:
        vals.append(100 - df["CPU_percent"].iloc[-1])
    if "Memory_Used_Percent" in df:
        vals.append(100 - df["Memory_Used_Percent"].iloc[-1])
    if "Disk_Used_Percent" in df:
        vals.append(100 - df["Disk_Used_Percent"].iloc[-1])

    return round(np.mean(vals),1) if vals else 100

def status_label(score):
    if score >= 80:
        return "🟢 Healthy"
    elif score >= 60:
        return "🟡 Warning"
    else:
        return "🔴 Critical"

def prediction_confidence(df):
    if len(df) < SEQ_LEN:
        return 0

    vals = df["CPU_percent"].dropna().tail(SEQ_LEN)

    if len(vals) < 10:
        return 50

    mean = vals.mean()
    std = vals.std()

    if mean == 0:
        return 50

    return round(max(0,100-(std/mean)*100),1)

def get_cutoff(window):
    num = int(window.split()[0])

    if "sec" in window:
        return datetime.now() - timedelta(seconds=num)
    elif "min" in window:
        return datetime.now() - timedelta(minutes=num)
    elif "hr" in window:
        return datetime.now() - timedelta(hours=num)
    elif "day" in window:
        return datetime.now() - timedelta(days=num)

    return datetime.now()

# =========================
# DATA
# =========================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

new = scrape()

# ensure columns exist
for col in ["CPU_percent","Memory_Used_Percent","Disk_Used_Percent","node1","node5","node15"]:
    if col not in new:
        new[col] = np.nan

st.session_state.df = pd.concat(
    [st.session_state.df, pd.DataFrame([new])],
    ignore_index=True
)


# =========================
# CONTROLS
# =========================
with st.container(border=True):
    st.subheader("⚙️ Control Panel")
    c1, c2 = st.columns(2)
    with c1:
        refresh = st.number_input("Refresh (sec)", step=1, min_value=1, value=5)
        st_autorefresh(interval=refresh*1000)
    with c2:
        window = st.selectbox(
            "Window",
            [
                "30 sec",
                "1 min", "5 min", "15 min", "30 min",
                "1 hr", "6 hr", "12 hr",
                "1 day", "2 days", "7 days"
            ],
            index=1  # 👉 default = 5 sec
        )

cutoff = get_cutoff(window)

# 🔥 FULL DATA (for prediction)
df_full = st.session_state.df.copy()

# 🔥 FILTERED DATA (for charts only)
df = df_full[df_full["timestamp"] >= cutoff]
df = df.tail(300)

if df.empty:
    st.warning("No data available in selected time window")
    st.stop()

# st_autorefresh(interval=refresh*1000)

# =========================
# SYSTEM OVERVIEW
# =========================
with st.container():
    st.subheader("📌 System Overview")

    prom_ok = check_prometheus()
    usable = len(df_full["CPU_percent"].dropna())

    c1,c2,c3,c4,c5 = st.columns(5, border=True)

    if usable >= SEQ_LEN:

        cpu_pred = predict(df_full["CPU_percent"], "cpu")

        health = calculate_health(df_full)

        if len(cpu_pred) == 0:
            future = health
        else:
            future = max(0, 100 - float(cpu_pred[-1]))

        conf = prediction_confidence(df_full)

        c1.metric("Health", f"{health}/100")
        c2.metric("Predicted Health", f"{future:.1f}/100", f"{future-health:+.1f}")
        c3.metric("Confidence", f"{conf}%")
        c4.metric("Status", status_label(health))
        c5.metric("Prometheus", "Connected" if prom_ok else "Disconnected ❌")

    else:
        c1.metric("Health","--")
        c2.metric("Prediction","--")
        c3.metric("Samples Needed", SEQ_LEN-usable)
        c4.metric("Status","Collecting")
        c5.metric("Prometheus", "Connected" if prom_ok else "Disconnected ❌")

# =========================
# PANEL
# =========================
def panel(df, col, key, title, is_percent=False):

    with st.container(border=True):

        st.subheader(title)

        current = df[col].iloc[-1]
        pred = predict(st.session_state.df[col], key)

        suffix = "%" if is_percent else ""

        c1,c2,c3 = st.columns(3, border=True)

        c1.metric("Current", f"{current:.2f}{suffix}")

        if len(pred) > 0:
            c2.metric("Next", f"{pred[0]:.2f}{suffix}", f"{pred[0]-current:+.2f}")
            c3.metric("Future", f"{pred[-1]:.2f}{suffix}", f"{pred[-1]-current:+.2f}")
        else:
            c2.metric("Next","--")
            c3.metric("Future","--")

        actual = df[["timestamp", col]].dropna()

        if len(pred) > 0:
            last = df["timestamp"].iloc[-1]

            future_time = [
                last + timedelta(seconds=SCRAPE_INTERVAL*(i+1))
                for i in range(FUTURE_STEPS)
            ]

            pred_df = pd.DataFrame({
                "timestamp": future_time,
                "Predicted": pred
            }).set_index("timestamp")

            actual = actual.rename(columns={col:"Actual"}).set_index("timestamp")

            merged = actual.merge(pred_df, how="outer", left_index=True, right_index=True)

            st.area_chart(merged)
        else:
            st.area_chart(actual.set_index("timestamp"))

# =========================
# DASHBOARD
# =========================
st.subheader("📈 Metrics Dashboard")

metrics = [
    ("CPU_percent","cpu","CPU Usage",True),
    ("Memory_Used_Percent","memory","Memory Usage",True),
    ("Disk_Used_Percent","disk","Disk Usage",True),
    ("node1","node1","Load 1",False),
    ("node5","node5","Load 5",False),
    ("node15","node15","Load 15",False),
]

for i in range(0, len(metrics), 2):
    cols = st.columns(2)

    for j in range(2):
        if i+j < len(metrics):
            col,key,title,p = metrics[i+j]
            with cols[j]:
                panel(df, col, key, title, p)

# =========================
# TABLE
# =========================
with st.container(border=True):
    st.subheader("📋 Latest Metrics")
    st.dataframe(df.tail(20), use_container_width=True)