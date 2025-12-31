## Problem Statement
Aviation safety teams receive large numbers of incident reports written in plain text
(e.g., bird strikes, turbulence, runway issues).

Manually reading and categorizing these reports is slow and inefficient.

**Goal of this project:**  
To demonstrate how a system can automatically:
- read incident descriptions (text),
- classify the **type of incident**,
- estimate the **severity level**,
- and show results **live on a dashboard**.

---

## What This Project Does
This project simulates aviation incident reports and processes them **one by one**, just
like real-time data.

In simple terms:
- A fake (synthetic) incident report is generated
- A machine learning model reads the text
- The model predicts:
  - **Incident Type** (Bird Strike, Turbulence, Runway Incursion, etc.)
  - **Severity** (Low / Medium / High)
- The dashboard updates instantly with metrics and charts

This shows how **real-time analytics and monitoring** would work in practice.

---

## How the System Works (Step-by-Step)

### 1️⃣ Incident Simulation
The app generates **synthetic incident narratives**, such as:
- “bird strike during climb”
- “severe turbulence at FL350”

These are **not real incidents** — they are created only for demonstration.

---

### 2️⃣ Text Processing (NLP)
Computers cannot understand text directly.

So the project uses **TF-IDF** to:
- convert text into numbers,
- identify important words,
- prepare data for machine learning.

---

### 3️⃣ Machine Learning Models
Two separate models are used:
- **Incident Type Classifier**
- **Severity Classifier**

Both models use **Logistic Regression**, which is:
- fast,
- easy to understand,
- suitable for text classification tasks.

---

### 4️⃣ Real-Time Simulation
Reports are processed **one at a time**:
- predictions are made instantly,
- results are stored in session memory,
- metrics update live.

This simulates how a streaming system behaves.

---

### 5️⃣ Live Dashboard & Monitoring
The Streamlit dashboard shows:
- number of reports processed,
- prediction accuracy,
- confusion matrices (using Plotly),
- details of the latest simulated report.

This demonstrates **real-time model monitoring**, not offline analysis.

---
# ✈️ ASRS Real-Time Incident Classification Simulator (Simulation)
---
## Tech Stack
- Python  
- Streamlit  
- pandas  
- scikit-learn  
- Plotly  

---

## ❗ Important Note (Read First)
This project **does NOT predict real or current airplane incidents**.  
It is a **real-time simulation** created to demonstrate how machine learning models
*would work* if aviation incident reports arrived live.

There is **no connection to real aircraft, airports, or live aviation data**.

## How to Run the Project

```bash
pip install streamlit pandas scikit-learn plotly
streamlit run app.py
