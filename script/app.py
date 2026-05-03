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
st.set_page_config(
    layout="wide",
    page_title="AI Infrastructure Monitoring Dashboard",
    page_icon="📊"
)
st.title("AI Infrastructure Monitoring Dashboard", text_alignment="center")
st.divider(width="stretch")

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
    vals = df["CPU_percent"].dropna()

    if len(vals) < 5:
        return 50   # start neutral instead of 0

    vals = vals.tail(min(len(vals), SEQ_LEN))

    mean = vals.mean()
    std = vals.std()

    if mean == 0:
        return 50

    return round(max(0, 100 - (std / mean) * 100), 1)

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
# CONFIDENCE HISTORY
# =========================
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = pd.DataFrame(columns=["timestamp", "confidence"])


# =========================
# CONTROLS
# =========================
with st.container():
    st.subheader("⚙️ Control Panel")

    c1, c2, c3 = st.columns(3, border=True)

    with c1:
        refresh = st.number_input("Refresh (sec)", step=1, min_value=1, value=5)

    with c2:
        window = st.selectbox(
            "Window",
            [
                "30 sec",
                "1 min", "5 min", "15 min", "30 min",
                "1 hr", "6 hr", "12 hr",
                "1 day", "2 days", "7 days"
            ],
            index=2
        )

    with c3:
        selected_metrics = st.multiselect(
            "Select Metrics",
            [
                "CPU Usage",
                "Memory Usage",
                "Disk Usage",
                "Load 1",
                "Load 5",
                "Load 15"
            ],
            default=[
                "CPU Usage",
                "Memory Usage",
                "Disk Usage"
            ]
        )

cutoff = get_cutoff(window)
st_autorefresh(interval=refresh*1000)

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

    c1, c2, c3, c4, c5 = st.columns(5, border=True)

    health = calculate_health(df_full)

    # =========================
    # BEFORE MODEL READY
    # =========================
    if usable < SEQ_LEN:

        c1.metric("❤️ Current Health", f"{health}/100", f"+/-{0} than Predicted Health", delta_arrow="off", delta_color="off")

        c2.metric(
            "💖 Predicted Health",
            "...",
            "Waiting for LSTM engine to start prediction",
            delta_color="off",
            delta_arrow="off"
        )

        c3.metric(
            "🧪 Samples Needed",
            SEQ_LEN - usable,
            "out of 40",
            delta_color="off",
            delta_arrow="off"
        )

        c4.metric("📋 Status", "Collecting Metrics")
        c5.metric("🔥 Prometheus", "Connected" if prom_ok else "Disconnected ❌")

    # =========================
    # AFTER MODEL READY
    # =========================
    else:

        # ---------- Prediction ----------
        cpu_pred = predict(df_full["CPU_percent"], "cpu")

        if len(cpu_pred) == 0:
            future = health
        else:
            future = max(0, 100 - float(cpu_pred[-1]))

        # ---------- Confidence ----------
        conf = prediction_confidence(df_full)

        # store ONLY after prediction starts
        st.session_state.confidence_history = pd.concat([
            st.session_state.confidence_history,
            pd.DataFrame([{
                "timestamp": datetime.now(),
                "confidence": conf
            }])
        ], ignore_index=True)

        # ---------- Prepare Sparkline ----------
        conf_df = st.session_state.confidence_history.copy()

        # apply same window
        conf_df = conf_df[conf_df["timestamp"] >= cutoff]

        conf_series = conf_df["confidence"].astype(float).tolist()

        # ✅ CRITICAL FIX: ensure sparkline always renders
        if len(conf_series) == 0:
            conf_series = [conf, conf]
        elif len(conf_series) == 1:
            conf_series = conf_series * 2

        # optional: slight smoothing for better UX
        if len(conf_series) > 3:
            conf_series = pd.Series(conf_series).rolling(3, min_periods=1).mean().tolist()

        # ---------- Metrics ----------
        c1.metric(
            "❤️ Current Health",
            f"{health}/100",
            f"{health - future:+.1f} than Predicted Health"
        )

        c2.metric(
            "💖 Predicted Health",
            f"{future:.1f}/100",
            f"{future - health:+.1f} than Current Health"
        )

        c3.metric(
            "🌟 Confidence",
            f"{conf}%",
            delta=None,
            chart_data=conf_series,
            chart_type="area"
        )

        c4.metric("📋 Status", status_label(health))
        c5.metric("🔥 Prometheus", "Connected" if prom_ok else "Disconnected ❌")
# =========================
# PANEL
# =========================
def panel(df, col, key, title, is_percent=False):

    with st.container(border=True):

        st.subheader(title, text_alignment="center", divider="grey")

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
                last + timedelta(seconds=SCRAPE_INTERVAL*i)
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

metrics = {
    "CPU Usage": ("CPU_percent","cpu",True),
    "Memory Usage": ("Memory_Used_Percent","memory",True),
    "Disk Usage": ("Disk_Used_Percent","disk",True),
    "Load 1": ("node1","node1",False),
    "Load 5": ("node5","node5",False),
    "Load 15": ("node15","node15",False),
}

selected_data = [
    (metrics[name][0], metrics[name][1], name, metrics[name][2])
    for name in selected_metrics
]

for i in range(0, len(selected_data), 2):
    cols = st.columns(2)

    for j in range(2):
        if i + j < len(selected_data):
            col, key, title, p = selected_data[i + j]
            with cols[j]:
                panel(df, col, key, title, p)

# =========================
# TABLE
# =========================
with st.container(border=True):
    st.subheader("📋 Latest Metrics")
    st.dataframe(df, width="stretch")