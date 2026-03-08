---
title: EV Battery Management Dashboard
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Explainable AI EV Battery Management System (EV-BMS)

An Explainable AI powered Electric Vehicle Battery Management Dashboard that integrates deep learning, reinforcement learning, and explainable AI to optimise EV battery health and charging decisions.

The system predicts Battery State-of-Health (SoH) using a GRU neural network and determines optimal charging actions using a Double Deep Q Network (Double DQN) while explaining the decisions using SHAP explainability.

---

# System Pipeline

Battery Sensor Inputs → GRU SoH Prediction → SoH Calibration → RL State Generation → Double DQN Charging Decision → SHAP Explainability → Dashboard Visualization

---

# Input Parameters

The dashboard collects real battery sensor measurements:

• IR — Internal Resistance  
• QC — Charge Capacity  
• QD — Discharge Capacity  
• Tavg — Average Temperature  
• Tmax — Maximum Temperature  
• Charge Time — Charging duration

These simulate real EV battery sensor data.

---

# GRU Model – SoH Prediction

The GRU model predicts battery State-of-Health using the following features:

IR  
QC  
QD  
Tavg  
Tmax  
ChargeTime

Input tensor shape:

(1, 20, 6)

Scaling files used:

gru_scaler_X.pkl  
gru_scaler_y.pkl

Output:

Predicted Battery State-of-Health (SoH)

---

# SoH Calibration

Battery degradation is estimated using cycle life.

cycle = int((1 − QD / QC) × 800)

degradation = 1 − (cycle / 800)

Final calibrated SoH:

SoH_final = Raw_SoH × (0.7 + 0.3 × degradation)

Range constraint:

0.5 ≤ SoH_final ≤ 1.0

---

# Battery Health Classification

SoH > 0.90 → Healthy  
0.80 – 0.90 → Moderate  
0.70 – 0.80 → Degrading  
< 0.70 → Severely Degraded

---

# Reinforcement Learning Controller

The Double DQN controller determines optimal charging actions.

RL State Vector:

[SoH, Temp, Cycle, Current]

Where:

Temp = Tavg  
Cycle = int((1 − QD / QC) × 800)  
Current = QC / (ChargeTime / 3600)

---

# Charging Actions

The RL agent predicts three actions:

Decrease Charging  
Maintain Charging  
Increase Charging

The action with the highest Q-value is selected.

---

# Explainable AI

SHAP is used to interpret reinforcement learning decisions.

Generated XAI visualisations include:

Global Feature Importance  
SHAP Distribution Plot  
SHAP Heatmap  
Temperature Dependence Plot  
Cycle Influence Plot  
Current Influence Plot  
RL Action Influence Plot  
Feature Ranking Plot  
Combined GRU + RL Explanation

These plots explain why the AI selected a particular charging strategy.

---

# Dashboard Components

Battery Health Panel  
• Predicted SoH  
• Battery health status  
• Interactive gauge

Charging Decision Panel  
• RL charging recommendation  
• Q-value comparison chart

Explainability Panel  
• SHAP visualisations explaining decisions

---

# Technology Stack

Streamlit  
TensorFlow  
NumPy  
Pandas  
Matplotlib  
Plotly  
SHAP  
Scikit-Learn  
Joblib

---

# Project Structure

```
.
├ app.py
├ README.md
├ requirements.txt
│
├ assets
│   └ style.css
│
├ models
│   ├ gru_soh_model.keras
│   ├ double_dqn_calibrated.keras
│   ├ gru_scaler_X.pkl
│   └ gru_scaler_y.pkl
│
└ utils
    ├ inference.py
    └ xai.py
```

---

# Run Locally

Install dependencies:

pip install -r requirements.txt

Run dashboard:

streamlit run app.py

---

# Deployment

The application is deployed using Hugging Face Spaces with Streamlit.

---

# Author

Vijay Manikanta  
B.Tech – Artificial Intelligence & Machine Learning  
Research Area: Explainable AI for Electric Vehicle Battery Management