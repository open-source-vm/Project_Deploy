# Explainable AI EV Battery Management Dashboard

A professional Streamlit dashboard integrating **GRU** SoH prediction, **cycle-calibrated SoH**, **Double DQN** charging optimisation, and **SHAP explainability** — with a dark neumorphism UI.

---

## Pipeline

```
Sensor Inputs (IR, QC, QD, Tavg, Tmax, ChargeTime)
→ GRU Raw SoH → Cycle Calibration → Health Classification
→ Auto RL State [SoH, Temp, Cycle, Current]
→ Double DQN Decision → SHAP Explainability → Dashboard
```

## User Inputs (sensor data only)

| Input | Description |
|-------|-------------|
| IR | Internal Resistance (Ω) |
| QC | Charge Capacity (Ah) |
| QD | Discharge Capacity (Ah) |
| Tavg | Average Temperature (°C) |
| Tmax | Maximum Temperature (°C) |
| ChargeTime | Charge duration (s) |

> SoH, Cycle, and Current are **never entered manually** — always derived automatically.

## Auto-derived RL State

| Parameter | Formula |
|-----------|---------|
| SoH | GRU prediction, cycle-calibrated, clamped [0.5, 1.0] |
| Temperature | Tavg |
| Cycle | `int((1 − QD/QC) × 800)` |
| Current | `QC / (ChargeTime / 3600)` |

## SoH Calibration

```
degradation = 1 − (cycle / 800)
SoH_final = raw_soh × (0.7 + 0.3 × degradation)     clamped [0.5, 1.0]
```

## Health Classification

| SoH | Status |
|-----|--------|
| > 0.90 | Healthy |
| 0.80–0.90 | Moderate |
| 0.70–0.80 | Degrading |
| < 0.70 | Severely Degraded |

## 9 SHAP Visualisations

Feature Importance · Distribution · Heatmap · Temperature · Cycle · Current · Action Influence · Feature Ranking · Combined GRU+RL

## Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Streamlit · TensorFlow · Plotly · SHAP · NumPy · Pandas · Matplotlib · Joblib · Scikit-learn
