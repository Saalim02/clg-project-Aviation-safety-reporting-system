"""
Streamlit ASRS Real-Time Simulation — Cleaned UI Version
- Responsive layout with sidebar controls, Plotly confusion matrices
- Flight numbers editable in sidebar and CSV upload option
- Auto-run with progress bar and data export

Save as `app.py` and run: `streamlit run app.py`
"""

import uuid
import time
import json
import random
from typing import Tuple, List

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.graph_objects as go

st.set_page_config(page_title="ASRS Live Simulator", layout="wide", page_icon="✈️")

# Small CSS for a cleaner look
st.markdown(
    """
    <style>
    .big-font {font-size:28px; font-weight:700;}
    .muted {color: #6b7280}
    .card {background: linear-gradient(180deg,#fff,#f7fafc); padding:16px; border-radius:10px; box-shadow:0 4px 20px rgba(16,24,40,0.06);} 
    </style>
    """,
    unsafe_allow_html=True,
)

TRAIN_SAMPLES = [
    ("bird strike during climb at 2500 ft", "Bird Strike", "Medium"),
    ("severe turbulence at fl350 caused minor injuries", "Turbulence", "High"),
    ("runway incursion detected vehicle crossing hold point", "Runway Incursion", "High"),
    ("engine vibration monitored precautionary landing", "Engine Anomaly", "Medium"),
]

texts = [t for t, _, _ in TRAIN_SAMPLES]
labels_type = [c for _, c, _ in TRAIN_SAMPLES]
labels_sev = [s for _, _, s in TRAIN_SAMPLES]

@st.cache_resource
def build_models():
    pipe_type = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=400))
    ])
    pipe_type.fit(texts, labels_type)

    pipe_sev = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=400))
    ])
    pipe_sev.fit(texts, labels_sev)
    return pipe_type, pipe_sev

pipe_type, pipe_sev = build_models()

INCIDENTS = [
    ("bird strike during climb", "Bird Strike", "Medium"),
    ("runway incursion", "Runway Incursion", "High"),
    ("severe turbulence at FL350", "Turbulence", "High"),
    ("engine vibration detected", "Engine Anomaly", "Medium"),
]

LABELS_TYPE = sorted(list({t[1] for t in INCIDENTS}))
LABELS_SEV = sorted(list({t[2] for t in INCIDENTS}))

# Session state
if "preds_type" not in st.session_state:
    st.session_state.preds_type = []
if "labels_type" not in st.session_state:
    st.session_state.labels_type = []
if "preds_sev" not in st.session_state:
    st.session_state.preds_sev = []
if "labels_sev" not in st.session_state:
    st.session_state.labels_sev = []
if "last_report" not in st.session_state:
    st.session_state.last_report = {}
if "auto_running" not in st.session_state:
    st.session_state.auto_running = False
if "flights" not in st.session_state:
    st.session_state.flights = ["DTK523", "AI101", "BA305", "LH789"]

# Helpers

def random_location() -> Tuple[float, float]:
    return round(random.uniform(-90, 90), 4), round(random.uniform(-180, 180), 4)


def classify(narrative: str):
    inc = pipe_type.predict([narrative])[0]
    sev = pipe_sev.predict([narrative])[0]
    return {"incident_type": inc, "severity": sev}


def generate_report(flights: List[str]) -> dict:
    sim_text, true_type, true_sev = random.choice(INCIDENTS)
    flight_number = random.choice(flights) if flights else "UNKNOWN"
    pred = classify(sim_text)

    st.session_state.preds_type.append(pred["incident_type"])
    st.session_state.labels_type.append(true_type)
    st.session_state.preds_sev.append(pred["severity"])
    st.session_state.labels_sev.append(true_sev)

    lat, lon = random_location()
    report_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "report_id": report_id,
        "flight_number": flight_number,
        "timestamp": timestamp,
        "narrative": sim_text,
        "incident_type": pred["incident_type"],
        "severity": pred["severity"],
        "lat": lat,
        "lon": lon,
    }
    st.session_state.last_report = payload
    return payload


def compute_metrics():
    total = len(st.session_state.labels_type)
    acc_type = accuracy_score(st.session_state.labels_type, st.session_state.preds_type) if total > 0 else 0.0
    acc_sev = accuracy_score(st.session_state.labels_sev, st.session_state.preds_sev) if total > 0 else 0.0
    return total, acc_type, acc_sev


def plot_confusion_plotly(actual, preds, labels, title):
    cm = confusion_matrix(actual, preds, labels=labels)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        hoverongaps=False,
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", margin=dict(t=40, b=10))
    return fig

# Sidebar controls
with st.sidebar:
    st.markdown("<div class='big-font'>✈️ ASRS Live — Controls</div>", unsafe_allow_html=True)
    st.write("Configure flights, auto-run, and exports.")
    st.markdown("---")

    flights_input = st.text_input("Flight numbers (comma-separated)", value=", ".join(st.session_state.flights))
    st.session_state.flights = [f.strip() for f in flights_input.split(",") if f.strip()]

    uploaded = st.file_uploader("Or upload CSV with column 'flight'", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "flight" in df_up.columns:
                st.session_state.flights = df_up["flight"].dropna().astype(str).tolist()
                st.success(f"Loaded {len(st.session_state.flights)} flights from CSV")
            else:
                st.warning("CSV missing 'flight' column")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("---")
    auto = st.checkbox("Enable auto-run", value=False, key="auto_checkbox")
    n_steps = st.number_input("Number of reports", min_value=1, max_value=2000, value=25)
    delay = st.slider("Delay (seconds)", min_value=0.1, max_value=3.0, value=0.8, step=0.1)

    st.markdown("---")
    if st.download_button("Download session JSON", json.dumps({
        "labels_type": st.session_state.labels_type,
        "preds_type": st.session_state.preds_type,
        "labels_sev": st.session_state.labels_sev,
        "preds_sev": st.session_state.preds_sev,
    }), file_name="asrs_session.json"):
        pass

    if st.button("Clear history"):
        st.session_state.preds_type.clear()
        st.session_state.labels_type.clear()
        st.session_state.preds_sev.clear()
        st.session_state.labels_sev.clear()
        st.session_state.last_report = {}
        st.success("History cleared")

# Main layout
st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>" +
            "<div class='big-font'>ASRS Real-Time Simulator</div>" +
            "<div class='muted'>Demo · Live metrics · Confusion matrices</div></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Controls")
    total, acc_type, acc_sev = compute_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Type acc", f"{acc_type*100:.2f}%")
    c3.metric("Severity acc", f"{acc_sev*100:.2f}%")

    if st.button("Generate single report"):
        payload = generate_report(st.session_state.flights)
        st.success("Report generated")
        st.json(payload)

    if auto and st.button("Start auto-run"):
        st.session_state.auto_running = True

    if st.session_state.auto_running and st.button("Stop auto-run"):
        st.session_state.auto_running = False

    prog = st.progress(0)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Last report")
    if st.session_state.last_report:
        last = st.session_state.last_report
        st.markdown(f"**Flight:** `{last['flight_number']}`  ")
        st.markdown(f"**Timestamp:** {last['timestamp']}  ")
        st.markdown(f"**Narrative:** {last['narrative']}  ")
        st.markdown(f"**Predicted:** {last['incident_type']} · {last['severity']}  ")
        st.write(last)
    else:
        st.info("No reports yet — generate one or start auto-run")
    st.markdown("</div>", unsafe_allow_html=True)

# Auto-run loop
if st.session_state.auto_running:
    steps = n_steps
    for i in range(steps):
        if not st.session_state.auto_running:
            break
        generate_report(st.session_state.flights)
        prog.progress((i+1)/steps)
        time.sleep(delay)
    st.session_state.auto_running = False
    st.experimental_rerun()

# Visualizations
st.markdown("---")
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("Confusion Matrix — Type")
    total, _, _ = compute_metrics()
    if total > 0:
        fig_type = plot_confusion_plotly(st.session_state.labels_type, st.session_state.preds_type, LABELS_TYPE, "Type")
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("Generate reports to populate confusion matrix")

with vis_col2:
    st.subheader("Confusion Matrix — Severity")
    if total > 0:
        fig_sev = plot_confusion_plotly(st.session_state.labels_sev, st.session_state.preds_sev, LABELS_SEV, "Severity")
        st.plotly_chart(fig_sev, use_container_width=True)
    else:
        st.info("Generate reports to populate confusion matrix")

# Training samples and notes
st.markdown("---")
notes_col1, notes_col2 = st.columns([1,2])

with notes_col1:
    st.subheader("Training samples")
    st.table(pd.DataFrame(TRAIN_SAMPLES, columns=["text","type","severity"]))

with notes_col2:
    st.subheader("Notes & Next steps")
    st.markdown("- Training dataset is toy-sized — expand for realistic performance")
    st.markdown("- Consider saving trained pipelines with joblib to avoid retrain on start")
    st.markdown("- Use a backend (FastAPI) for high-frequency real-time streaming")

# Export CSV
st.markdown("---")
if st.button("Download all session reports as CSV"):
    df = pd.DataFrame({
        "pred_type": st.session_state.preds_type,
        "label_type": st.session_state.labels_type,
        "pred_sev": st.session_state.preds_sev,
        "label_sev": st.session_state.labels_sev,
    })
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="asrs_reports.csv")
