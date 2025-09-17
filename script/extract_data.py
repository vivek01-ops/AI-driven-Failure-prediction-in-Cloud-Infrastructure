import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIGURATION ----------------
PROMETHEUS_BASE = "http://192.168.49.2:32600/api/v1"
PROMETHEUS_URL = f"{PROMETHEUS_BASE}/query"
SCRAPE_INTERVAL = 10  # seconds

# Metrics we want
METRICS_RAW = {
    "CPU_percent": '100 - (avg by (instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
    "node_memory_MemAvailable_bytes": "node_memory_MemAvailable_bytes",
    "node_memory_MemTotal_bytes": "node_memory_MemTotal_bytes",
    "node_filesystem_size_bytes": 'avg(node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})',
    "node_filesystem_avail_bytes": 'avg(node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"})'
}

NODE_LOAD_METRICS = ["node_load1", "node_load5", "node_load15"]


# ---------------- FUNCTIONS ----------------
def query_prometheus(query):
    """Query a single metric and return its numeric value"""
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=3)
        result = response.json()
        if result.get("status") == "success":
            values = result["data"]["result"]
            if values:
                vals = [float(v["value"][1]) for v in values]
                return sum(vals) / len(vals)
        return None
    except Exception:
        return None


def get_node_loads():
    """Fetch node_load1, node_load5, node_load15 per node"""
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
    """Scrape only selected metrics"""
    data = {"timestamp": datetime.now()}

    # Core metrics
    for name, query in METRICS_RAW.items():
        value = query_prometheus(query)
        if value is not None:
            data[name] = value

    # Node load metrics
    data.update(get_node_loads())
    return data


def process_metrics(df):
    """Add calculated Memory and Disk usage"""
    if df.empty:
        return df

    # --- Memory ---
    if "node_memory_MemAvailable_bytes" in df.columns and "node_memory_MemTotal_bytes" in df.columns:
        df["Memory_GiB_Available"] = df["node_memory_MemAvailable_bytes"] / (1024**3)
        df["Memory_GiB_Total"] = df["node_memory_MemTotal_bytes"] / (1024**3)
        df["Memory_GiB_Used"] = df["Memory_GiB_Total"] - df["Memory_GiB_Available"]
        df["Memory_Used_Percent"] = (df["Memory_GiB_Used"] / df["Memory_GiB_Total"]) * 100

    # --- Disk ---
    if "node_filesystem_size_bytes" in df.columns and "node_filesystem_avail_bytes" in df.columns:
        df["Disk_GiB_Total"] = df["node_filesystem_size_bytes"] / (1024**3)
        df["Disk_GiB_Available"] = df["node_filesystem_avail_bytes"] / (1024**3)
        df["Disk_GiB_Used"] = df["Disk_GiB_Total"] - df["Disk_GiB_Available"]
        df["Disk_Used_Percent"] = (df["Disk_GiB_Used"] / df["Disk_GiB_Total"]) * 100

    return df


# ---------------- STREAMLIT APP ----------------
st.set_page_config(layout="wide")
st.title("Real-Time Prometheus Dashboard (CPU %, Memory, Disk, Node Loads)")

# Sidebar / Controls
col1, col2 = st.columns(2, gap="medium")
with col1:
    interval = st.number_input("Scrape Interval (sec)", min_value=1, value=SCRAPE_INTERVAL)
with col2:
    time_window_option = st.selectbox(
        "Time Window",
        ("30 sec", "1 min", "5 min", "10 min", "30 min",
         "1 hr", "2 hr", "5 hr", "6 hr", "8 hr", "12 hr", "18 hr", "24 hr")
    )

# Convert time window to timedelta
num, unit = time_window_option.split()
num = int(num)
if unit.startswith("sec"):
    time_delta = timedelta(seconds=num)
elif unit.startswith("min"):
    time_delta = timedelta(minutes=num)
else:
    time_delta = timedelta(hours=num)

st_autorefresh(interval=interval * 1000, key="auto_refresh")

# Initialize DataFrame
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["timestamp"] + list(METRICS_RAW.keys()) + NODE_LOAD_METRICS)

# Scrape metrics
new_row = scrape_metrics()

if not new_row:
    st.warning("No metrics scraped yet — waiting for Prometheus.")
    st.stop()

# Append new row
if st.session_state.df.empty or new_row != st.session_state.df.iloc[-1].to_dict():
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)

# Process metrics
df_display = process_metrics(st.session_state.df.copy())
time_window = datetime.now() - time_delta
df_recent = df_display[df_display["timestamp"] >= time_window]


# ---------------- DASHBOARD ----------------
cola, colb, colc = st.columns(3, border=True)

with cola:
    if "Memory_GiB_Used" in df_recent.columns:
        st.subheader(f"Memory Usage (GiB) ({time_window_option})", divider="red")
        st.area_chart(df_recent, x="timestamp", y=["Memory_GiB_Total", "Memory_GiB_Used"], use_container_width=True)

with colb:
    if "CPU_percent" in df_recent.columns:
        st.subheader(f"CPU Usage (%) ({time_window_option})", divider="red")
        st.line_chart(df_recent, x="timestamp", y="CPU_percent", use_container_width=True)

with colc:
    if "Disk_GiB_Used" in df_recent.columns:
        st.subheader(f"Disk Usage (GiB) ({time_window_option})", divider="red")
        st.area_chart(df_recent, x="timestamp", y=["Disk_GiB_Total", "Disk_GiB_Used"], use_container_width=True)

# Node load charts
load_columns = [col for col in df_recent.columns if "_node_load" in col]
if load_columns:
    with st.container(border=True):
        st.subheader(f"Node Load Averages ({time_window_option})", divider="red")

        node_map = {}
        for col in load_columns:
            if "_" in col:
                node, metric = col.split("_", 1)
                node_map.setdefault(node, []).append(col)

        for node, metrics in node_map.items():
            st.write(f"**Node: {node}**")
            st.line_chart(df_recent, x="timestamp", y=metrics, use_container_width=True)

# ---------------- DATA TABLE ----------------
st.subheader("Recent Data", divider="red")
rown = st.selectbox("Show last N rows", (10, 20, 30, 40, 50, 100))

# Show only selected useful columns in DataFrame
useful_cols = [c for c in df_display.columns if c in [
    "timestamp", "CPU_percent",
    "Memory_GiB_Total", "Memory_GiB_Used", "Memory_Used_Percent",
    "Disk_GiB_Total", "Disk_GiB_Used", "Disk_Used_Percent"
] or "_node_load" in c]

st.dataframe(df_display[useful_cols].tail(rown))

# ---------------- SAVE CLEAN CSV ----------------
csv_path = "data/cleaned_metrics1.csv"
try:
    df_display[useful_cols].to_csv(csv_path, index=False, float_format="%.3f")
except Exception:
    st.warning("Could not save CSV (check 'data/' folder exists).")
