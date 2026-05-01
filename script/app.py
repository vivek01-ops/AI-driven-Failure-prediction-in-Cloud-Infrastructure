import streamlit as st
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ======================================================
# CONFIG
# ======================================================
PROMETHEUS_BASE = "http://192.168.49.2:30477/api/v1"
PROM_URL = f"{PROMETHEUS_BASE}/query"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi4-mini"

SCRAPE_INTERVAL = 5
SEQ_LEN = 40
FUTURE_STEPS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"

# ======================================================
# PAGE
# ======================================================
st.set_page_config(
    page_title="AI Infrastructure Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Infrastructure Monitoring Dashboard")
st.caption(
    "Real-time Monitoring • LSTM Forecasting • Root Cause AI using Ollama"
)

# ======================================================
# MODEL
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
            torch.load(
                f"{MODEL_DIR}/{t}_model.pth",
                map_location=DEVICE
            )
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
    "cpu":
        '100 - (avg by(instance)'
        '(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',

    "mem_avail": "node_memory_MemAvailable_bytes",
    "mem_total": "node_memory_MemTotal_bytes",

    "disk_avail":
        'avg(node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"})',

    "disk_total":
        'avg(node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})',

    "node1": "avg(node_load1)",
    "node5": "avg(node_load5)",
    "node15": "avg(node_load15)",
}

# ======================================================
# HELPERS
# ======================================================
def query_prom(query):
    try:
        r = requests.get(
            PROM_URL,
            params={"query": query},
            timeout=3
        )

        vals = r.json()["data"]["result"]

        if vals:
            nums = [float(v["value"][1]) for v in vals]
            return np.mean(nums)

    except:
        return None

    return None


def get_cutoff(window):
    num = int(window.split()[0])

    if "sec" in window:
        return datetime.now() - timedelta(seconds=num)

    elif "min" in window:
        return datetime.now() - timedelta(minutes=num)

    elif "hr" in window:
        return datetime.now() - timedelta(hours=num)

    else:
        return datetime.now() - timedelta(days=num)


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
        row["Memory_Used_Percent"] = (
            (mem_total - mem_avail) / mem_total
        ) * 100

    if disk_avail and disk_total:
        row["Disk_Used_Percent"] = (
            (disk_total - disk_avail) / disk_total
        ) * 100

    if node1 is not None:
        row["node1"] = node1

    if node5 is not None:
        row["node5"] = node5

    if node15 is not None:
        row["node15"] = node15

    return row


def predict_metric(series, metric_name):

    if len(series) < SEQ_LEN:
        return []

    if metric_name not in MODELS:
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

    pred = scaler.inverse_transform(
        pred.reshape(-1, 1)
    ).flatten()

    st.write(f"{metric_name} len:", len(series))

    return pred


def health_status(score):
    if score >= 80:
        return "🟢 Healthy"
    elif score >= 60:
        return "🟡 Warning"
    else:
        return "🔴 Critical"


def calculate_health(df):

    parts = []

    if "CPU_percent" in df.columns:
        parts.append(100 - df["CPU_percent"].iloc[-1])

    if "Memory_Used_Percent" in df.columns:
        parts.append(
            100 - df["Memory_Used_Percent"].iloc[-1]
        )

    if "Disk_Used_Percent" in df.columns:
        parts.append(
            100 - df["Disk_Used_Percent"].iloc[-1]
        )

    for col in ["node1", "node5", "node15"]:

        if col in df.columns:

            cur = df[col].iloc[-1]
            avg = max(df[col].mean(), 1)

            parts.append(
                max(
                    0,
                    100 - ((cur / (avg * 2)) * 100)
                )
            )

    return round(np.mean(parts), 1) if parts else 100


def calculate_predicted_health(df):

    scores = []

    metric_map = {
        "CPU_percent": "cpu",
        "Memory_Used_Percent": "memory",
        "Disk_Used_Percent": "disk",
        "node1": "node1",
        "node5": "node5",
        "node15": "node15"
    }

    for col, key in metric_map.items():

        if col in df.columns:

            pred = predict_metric(
                df[col].dropna().tolist(),
                key
            )

            if len(pred) == 0:
                continue

            val = pred[-1]

            if col in [
                "CPU_percent",
                "Memory_Used_Percent",
                "Disk_Used_Percent"
            ]:
                scores.append(100 - val)

            else:
                avg = max(df[col].mean(), 1)

                scores.append(
                    max(
                        0,
                        100 - ((val / (avg * 2)) * 100)
                    )
                )

    return round(np.mean(scores), 1) if scores else 100


def calculate_prediction_confidence(df):

    if len(df) < SEQ_LEN:
        return 0

    scores = []

    for col in [
        "CPU_percent",
        "Memory_Used_Percent",
        "Disk_Used_Percent",
        "node1",
        "node5",
        "node15"
    ]:

        if col in df.columns:

            vals = df[col].dropna().tail(SEQ_LEN)

            if len(vals) < 10:
                continue

            mean = vals.mean()
            std = vals.std()

            if mean == 0:
                continue

            cv = std / mean
            score = max(0, 100 - (cv * 100))
            scores.append(min(score, 100))

    return round(np.mean(scores), 1) if scores else 75


# ======================================================
# OLLAMA AI
# ======================================================
def ask_ollama_rootcause(df, health, future_health, status):

    try:

        cpu = round(df["CPU_percent"].iloc[-1], 1) \
            if "CPU_percent" in df.columns else 0

        mem = round(df["Memory_Used_Percent"].iloc[-1], 1) \
            if "Memory_Used_Percent" in df.columns else 0

        disk = round(df["Disk_Used_Percent"].iloc[-1], 1) \
            if "Disk_Used_Percent" in df.columns else 0

        load1 = round(df["node1"].iloc[-1], 2) \
            if "node1" in df.columns else 0

        prompt = f"""
You are an expert DevOps AI assistant.

Analyze this infrastructure state:

CPU: {cpu}%
Memory: {mem}%
Disk: {disk}%
Load1: {load1}

Current Health Score: {health}/100
Predicted Health Score: {future_health}/100
Status: {status}

Return ONLY valid JSON:

{{
  "root_cause": "...",
  "impact": "...",
  "recommended_fix": "...",
  "urgency": "Low/Medium/High"
}}
"""

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=40
        )

        txt = r.json()["response"].strip()

        start = txt.find("{")
        end = txt.rfind("}") + 1

        txt = txt[start:end]

        return json.loads(txt)

    except Exception as e:

        return {
            "root_cause": "AI analysis unavailable",
            "impact": str(e),
            "recommended_fix": "Check Ollama service",
            "urgency": "Low"
        }


# ======================================================
# CHART
# ======================================================
def show_chart(df, col, metric_key, title):

    if col not in df.columns or df.empty:
        return

    st.markdown(f"#### {title}")

    actual = df[["timestamp", col]].copy()
    current_actual = df[col].iloc[-1]

    pred = predict_metric(
        df[col].dropna().tolist(),
        metric_key
    )

    suffix = "%" if metric_key in [
        "cpu", "memory", "disk"
    ] else ""

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Current",
            f"{current_actual:.2f}{suffix}",
            border=True
        )

    if len(pred) == 0:

        with b:
            st.metric("Next", "--", border=True)

        with c:
            st.metric("Forecast", "--", border=True)

        st.area_chart(
            actual.set_index("timestamp"),
            width='stretch'
        )
        return

    current_pred = pred[0]
    future_pred = pred[-1]

    delta1 = current_pred - current_actual
    delta2 = future_pred - current_actual

    last_time = df["timestamp"].iloc[-1]

    future_label = (
        last_time +
        timedelta(
            seconds=SCRAPE_INTERVAL * FUTURE_STEPS
        )
    ).strftime("%H:%M:%S")

    with b:
        st.metric(
            "Next",
            f"{current_pred:.2f}{suffix}",
            f"{delta1:+.2f}",
            border=True
        )

    with c:
        st.metric(
            future_label,
            f"{future_pred:.2f}{suffix}",
            f"{delta2:+.2f}",
            border=True
        )

    future_times = [
        last_time + timedelta(
            seconds=SCRAPE_INTERVAL * (i + 1)
        )
        for i in range(FUTURE_STEPS)
    ]

    pred_df = pd.DataFrame({
        "timestamp": future_times,
        "Predicted": pred
    })

    actual = actual.rename(
        columns={col: "Actual"}
    ).set_index("timestamp")

    pred_df = pred_df.set_index("timestamp")

    merged = actual.merge(
        pred_df,
        left_index=True,
        right_index=True,
        how="outer"
    )

    st.area_chart(
        merged,
        width='stretch'
    )


# ======================================================
# CONTROLS
# ======================================================
st.subheader("⚙️ Control Panel")

c1, c2, c3 = st.columns([1, 1, 2], border=True)

with c1:
    interval = st.selectbox(
        "Refresh (sec)",
        [1, 2, 5, 10],
        index=2
    )

with c2:
    time_window = st.selectbox(
        "Time Window",
        [
            "30 sec", "1 min", "5 min", "15 min",
            "30 min", "1 hr", "6 hr",
            "12 hr", "24 hr", "2 days"
        ],
        index=2
    )

with c3:
    selected = st.multiselect(
        "Metrics",
        [
            "CPU",
            "Memory",
            "Disk",
            "Load1",
            "Load5",
            "Load15"
        ],
        default=["CPU", "Memory", "Disk"]
    )

st_autorefresh(
    interval=interval * 1000,
    key="refresh"
)

# ======================================================
# DATA STORE
# ======================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

new_row = scrape_metrics()

st.session_state.df = pd.concat(
    [
        st.session_state.df,
        pd.DataFrame([new_row])
    ],
    ignore_index=True
)

cutoff = get_cutoff(time_window)

df = st.session_state.df.copy()
df = df[df["timestamp"] >= cutoff].copy()

prediction_ready = len(df) > SEQ_LEN - 1

# ======================================================
# SESSION STATE
# ======================================================
if "ai_rootcause" not in st.session_state:
    st.session_state.ai_rootcause = {
        "root_cause": "Collecting data...",
        "impact": "",
        "recommended_fix": "",
        "urgency": "Low"
    }

if "last_ai_run" not in st.session_state:
    st.session_state.last_ai_run = 0

# ======================================================
# SYSTEM OVERVIEW
# ======================================================
st.divider()
st.subheader("📌 System Overview")

k1, k2, k3, k4 = st.columns(4, border=True)

if prediction_ready:

    health = calculate_health(df)
    future_health = calculate_predicted_health(df)
    conf = calculate_prediction_confidence(df)
    status = health_status(health)

    now = time.time()

    if now - st.session_state.last_ai_run > 30:

        st.session_state.ai_rootcause = (
            ask_ollama_rootcause(
                df,
                health,
                future_health,
                status
            )
        )

        st.session_state.last_ai_run = now

    with k1:
        st.metric(
            "Current Health",
            f"{health}/100"
        )

    with k2:
        st.metric(
            f"Predicted +{SCRAPE_INTERVAL * FUTURE_STEPS}s",
            f"{future_health}/100",
            delta=f"{future_health-health:+.1f}"
        )

    with k3:
        st.metric(
            "Confidence",
            f"{conf}%"
        )

    with k4:
        st.metric(
            "Status",
            status
        )

else:

    remain = SEQ_LEN - len(df)

    with k1:
        st.metric("Health", "--")

    with k2:
        st.metric("Predicted", "--")

    with k3:
        st.metric("Samples Needed", remain)

    with k4:
        st.metric("Status", "Collecting")

# ======================================================
# AI ROOT CAUSE
# ======================================================
st.divider()
st.subheader("🤖 AI Root Cause Analysis")

x1, x2 = st.columns(2, border=True)

with x1:

    st.markdown("### 🔍 Root Cause")

    urgency = st.session_state.ai_rootcause["urgency"]

    msg = st.session_state.ai_rootcause["root_cause"]

    if urgency == "High":
        st.error(msg)

    elif urgency == "Medium":
        st.warning(msg)

    else:
        st.badge(msg)

    st.info(
        st.session_state.ai_rootcause["impact"]
    )

with x2:

    st.markdown("### 🛠 Recommended Fix")

    st.success(
        st.session_state.ai_rootcause[
            "recommended_fix"
        ]
    )

    st.metric(
        "Urgency",
        urgency
    )

# ======================================================
# CHARTS
# ======================================================
st.divider()
st.subheader("📈 Metrics Dashboard")

chart_map = {
    "CPU": (
        "CPU_percent",
        "cpu",
        "CPU Usage"
    ),

    "Memory": (
        "Memory_Used_Percent",
        "memory",
        "Memory Usage"
    ),

    "Disk": (
        "Disk_Used_Percent",
        "disk",
        "Disk Usage"
    ),

    "Load1": (
        "node1",
        "node1",
        "Load 1"
    ),

    "Load5": (
        "node5",
        "node5",
        "Load 5"
    ),

    "Load15": (
        "node15",
        "node15",
        "Load 15"
    ),
}

for i in range(0, len(selected), 2):

    row = selected[i:i+2]

    cols = st.columns(len(row), border=True)

    for j, metric in enumerate(row):

        col, key, title = chart_map[metric]

        with cols[j]:
            show_chart(
                df,
                col,
                key,
                title
            )

# ======================================================
# EXPORT
# ======================================================
st.divider()

left, right = st.columns([1, 2])

with left:
    st.download_button(
        "⬇ Export CSV",
        df.to_csv(index=False),
        file_name="metrics.csv",
        mime="text/csv",
        width='stretch'
    )

with right:
    st.info("Download visible dataset")

# ======================================================
# TABLE
# ======================================================
st.divider()
st.subheader("🗂 Latest Metrics")

st.dataframe(
    df.tail(20),
    width='stretch',
    height=420
)