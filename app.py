"""
app.py — Explainable AI EV Battery Management Dashboard
White Neumorphism (Soft UI) Theme

Pipeline:
    Sensor Inputs (IR, QC, QD, Tavg, Tmax, ChargeTime)
    → GRU Raw SoH → Cycle Calibration → Health Status
    → Auto RL State [SoH, Temp, Cycle, Current]
    → Double DQN Decision → SHAP Explainability → Dashboard
"""

import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from utils.inference import (
    load_models,
    prepare_gru_sequence,
    predict_soh,
    calibrate_soh,
    estimate_cycle,
    estimate_current,
    construct_rl_state,
    predict_rl_action,
    get_battery_health_label,
)
from utils.xai import (
    compute_shap_values,
    generate_feature_importance_plot,
    generate_shap_distribution_plot,
    generate_shap_heatmap,
    generate_temperature_dependence,
    generate_cycle_dependence,
    generate_current_dependence,
    generate_action_influence_plot,
    generate_feature_ranking_plot,
    generate_combined_xai_plot,
    generate_reasoning_text,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable AI EV-BMS Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar not needed now
)

# ── CSS ──────────────────────────────────────────────────────────────────────
def _load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
try:
    _load_css(os.path.join("assets", "style.css"))
except FileNotFoundError:
    pass

# ── Initialize Session State ─────────────────────────────────────────────────
defaults = {
    "ir": 0.04,
    "qc": 1.5,
    "qd": 1.5,
    "tavg": 30.0,
    "tmax": 35.0,
    "chargetime": 5000,
    "show_inputs": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER — title + subtitle
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="dashboard-title">'
    '⚡ Explainable AI EV Battery Management Dashboard'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="dashboard-subtitle">'
    'AI-driven battery health prediction and charging optimisation &nbsp;·&nbsp; '
    'GRU + Double DQN + SHAP'
    '</p>',
    unsafe_allow_html=True,
)

# ── ⚙ Battery Inputs Pure HTML Toggle (Uiverse.io) ───────────────────────────
_col_btn, _col_pad = st.columns([2, 5])
with _col_btn:
    toggle_html = f"""
    <style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
        display: flex;
        align-items: center;
        height: 100vh;
    }}
    .switch-container {{
      position: relative;
      width: 150px;
      height: 60px;
      background: #d6d6d6;
      border-radius: 50px;
      box-shadow:
        inset -8px -8px 16px #ffffff,
        inset 8px 8px 16px #b0b0b0;
    }}
    .toggle-checkbox {{
      display: none;
    }}
    .switch {{
      position: absolute;
      top: 50%;
      left: 0;
      width: 100%;
      height: 100%;
      transform: translateY(-50%);
      border-radius: 50px;
      overflow: hidden;
      cursor: pointer;
    }}
    .toggle {{
      position: absolute;
      width: 80px;
      height: 50px;
      background: linear-gradient(145deg, #d9d9d9, #bfbfbf);
      border-radius: 50px;
      top: 5px;
      left: 5px;
      box-shadow:
        -4px -4px 8px #ffffff,
        4px 4px 8px #b0b0b0;
      transition: all 0.3s ease-in-out;
      display: flex;
      align-items: center;
      justify-content: flex-start;
      padding-left: 10px;
    }}
    .led {{
      width: 10px;
      height: 10px;
      background: grey;
      border-radius: 50%;
      box-shadow: 0 0 10px 2px rgba(0, 0, 0, 0.2);
      transition: all 0.3s ease-in-out;
    }}
    .toggle-checkbox:checked + .switch .toggle {{
      left: 65px;
      background: linear-gradient(145deg, #cfcfcf, #a9a9a9);
      box-shadow:
        -4px -4px 8px #ffffff,
        4px 4px 8px #8a8a8a;
    }}
    .toggle-checkbox:checked + .switch .led {{
      background: yellow;
      box-shadow: 0 0 15px 4px yellow;
    }}
    .switch:hover .toggle {{
      box-shadow:
        -4px -4px 12px #ffffff,
        4px 4px 12px #9b9b9b;
    }}
    p.label-text {{
      font-family: 'Inter', sans-serif;
      font-size: 1.1rem;
      font-weight: 700;
      color: #475569;
      margin-left: 170px; /* Push past the 150px switch */
      position: absolute;
      white-space: nowrap;
      pointer-events: none;
    }}
    </style>
    
    <div class="switch-container">
      <input class="toggle-checkbox" id="toggle-switch" type="checkbox" {'checked' if st.session_state.show_inputs else ''} />
      <label class="switch" for="toggle-switch">
        <div class="toggle">
          <div class="led"></div>
        </div>
      </label>
      <p class="label-text">⚙ Battery Inputs</p>
    </div>
    
    <script>
      const checkbox = document.getElementById('toggle-switch');
      checkbox.addEventListener('change', function() {{
         // Send the checked state back to Streamlit
         window.parent.postMessage({{
             type: 'streamlit:setComponentValue',
             value: this.checked
         }}, '*');
      }});
    </script>
    """
    
    # Render the HTML component and capture its return value
    toggle_val = components.html(toggle_html, height=80)
    
    # If the user clicked the custom HTML toggle, update session state and rerun
    if toggle_val is not None and toggle_val != st.session_state.show_inputs:
        st.session_state.show_inputs = toggle_val
        st.rerun()

# ── Battery Sensor Inputs panel (collapsible in main page) ───────────────────
if st.session_state.show_inputs:
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">� Battery Sensor Inputs</div>',
        unsafe_allow_html=True,
    )

    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        st.session_state.ir = st.number_input(
            "Internal Resistance — IR (Ω)",
            min_value=0.01, max_value=0.08,
            value=float(st.session_state.ir), step=0.001,
            format="%.3f"
        )
        st.session_state.tavg = st.number_input(
            "Average Temperature — Tavg (°C)",
            min_value=10.0, max_value=50.0,
            value=float(st.session_state.tavg), step=0.5,
            format="%.1f"
        )
    with _c2:
        st.session_state.qc = st.number_input(
            "Charge Capacity — QC (Ah)",
            min_value=0.5, max_value=2.5,
            value=float(st.session_state.qc), step=0.01,
            format="%.2f"
        )
        st.session_state.tmax = st.number_input(
            "Maximum Temperature — Tmax (°C)",
            min_value=15.0, max_value=60.0,
            value=float(st.session_state.tmax), step=0.5,
            format="%.1f"
        )
    with _c3:
        st.session_state.qd = st.number_input(
            "Discharge Capacity — QD (Ah)",
            min_value=0.5, max_value=2.5,
            value=float(st.session_state.qd), step=0.01,
            format="%.2f"
        )
        st.session_state.chargetime = st.number_input(
            "Charge Time (seconds)",
            min_value=1000, max_value=10000,
            value=int(st.session_state.chargetime), step=100
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────────────────────
with st.spinner("Loading ML models …"):
    gru_model, dqn_model, scaler_X, scaler_y = load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# 1. GRU → raw SoH
sequence = prepare_gru_sequence(
    scaler_X,
    st.session_state.ir, st.session_state.qc, st.session_state.qd,
    st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime,
)
raw_soh = predict_soh(gru_model, scaler_y, sequence)

# 2. Auto-derive Cycle & Current
est_cycle = estimate_cycle(st.session_state.qd, st.session_state.qc)
est_current = estimate_current(st.session_state.qc, st.session_state.chargetime)

# 3. Calibrate SoH
cal_soh = calibrate_soh(raw_soh, est_cycle)
health_label = get_battery_health_label(cal_soh)

# 4. RL state + decision
rl_state = construct_rl_state(cal_soh, st.session_state.tavg, est_cycle, est_current)
action, q_values = predict_rl_action(dqn_model, rl_state)

action_map = {0: "decrease", 1: "maintain", 2: "increase"}
action_class = action_map.get(action, "maintain")
action_text_map = {
    0: "⬇  Decrease Charging",
    1: "⏸  Maintain Charging",
    2: "⬆  Increase Charging",
}
action_text = action_text_map.get(action, "Maintain Charging")

# 5. SHAP
shap_values, chosen_action, _ = compute_shap_values(dqn_model, rl_state)
gru_raw_inputs = [
    st.session_state.ir, st.session_state.qc, st.session_state.qd,
    st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime,
]


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT  — Row 1: Battery Health | Charging Decision
# ══════════════════════════════════════════════════════════════════════════════
col_h, col_a = st.columns([1, 1], gap="large")

# ── Battery Health Panel ─────────────────────────────────────────────────────
with col_h:
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔋 Battery Health</div>', unsafe_allow_html=True)

    cal_pct = cal_soh * 100
    raw_pct = raw_soh * 100

    st.markdown(f'<div class="soh-value">{cal_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="soh-label">Calibrated State-of-Health</div>', unsafe_allow_html=True)

    # Health badge
    badge_key = health_label.lower().replace(" ", "-")
    st.markdown(
        f'<div style="text-align:center;margin:0.5rem 0">'
        f'<span class="health-badge health-{badge_key}">{health_label}</span></div>',
        unsafe_allow_html=True,
    )

    # Inset metric boxes
    st.markdown(
        '<div class="soh-metrics">'
        f'<div class="soh-metric"><div class="metric-val">{raw_pct:.2f}%</div>'
        f'<div class="metric-lbl">Raw SoH (GRU)</div></div>'
        f'<div class="soh-metric"><div class="metric-val">{cal_pct:.2f}%</div>'
        f'<div class="metric-lbl">Calibrated SoH</div></div>'
        f'<div class="soh-metric"><div class="metric-val">{est_cycle}</div>'
        f'<div class="metric-lbl">Est. Cycle</div></div>'
        f'<div class="soh-metric"><div class="metric-val">{est_current:.2f}A</div>'
        f'<div class="metric-lbl">Est. Current</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Plotly semicircle gauge
    gauge_color = {
        "Healthy": "#22c55e",
        "Moderate": "#f59e0b",
        "Degrading": "#fb923c",
        "Severely Degraded": "#ef4444",
    }.get(health_label, "#f59e0b")

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cal_pct,
        number={"suffix": "%", "font": {"color": "#1e293b", "size": 24}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickwidth": 1,
                     "tickfont": {"color": "#64748b"}},
            "bar": {"color": gauge_color, "thickness": 0.3},
            "bgcolor": "#e2e8f0",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70],   "color": "rgba(239,68,68,0.08)"},
                {"range": [70, 80],  "color": "rgba(251,146,60,0.08)"},
                {"range": [80, 90],  "color": "rgba(245,158,11,0.08)"},
                {"range": [90, 100], "color": "rgba(34,197,94,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 2},
                "thickness": 0.8, "value": cal_pct,
            },
        },
    ))
    fig_g.update_layout(
        height=200, margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333"),
    )
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Charging Decision Panel ──────────────────────────────────────────────────
with col_a:
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙️ Charging Decision</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="action-badge {action_class}">{action_text}</div>',
        unsafe_allow_html=True,
    )

    # RL state pills
    st.markdown(
        '<div class="rl-pills">'
        f'<div class="rl-pill">SoH: <strong>{cal_soh:.4f}</strong></div>'
        f'<div class="rl-pill">Temp: <strong>{st.session_state.tavg:.1f} °C</strong></div>'
        f'<div class="rl-pill">Cycle: <strong>{est_cycle}</strong></div>'
        f'<div class="rl-pill">Current: <strong>{est_current:.2f} A</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Q-values bar chart
    fig_q = go.Figure(data=[go.Bar(
        x=["Decrease", "Maintain", "Increase"],
        y=q_values.tolist(),
        marker_color=["#ef4444", "#f59e0b", "#22c55e"],
        marker_line=dict(color="#c5cbd3", width=1),
        text=[f"{v:.4f}" for v in q_values],
        textposition="outside",
        textfont=dict(color="#333333", size=11),
    )])
    fig_q.update_layout(
        title=dict(text="Q-Values per Action", font=dict(color="#64748b", size=12)),
        height=260, margin=dict(l=30, r=30, t=40, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333"),
        yaxis=dict(gridcolor="#d1d5db", zerolinecolor="#d1d5db"),
        xaxis=dict(gridcolor="#d1d5db"),
        bargap=0.35,
    )
    st.plotly_chart(fig_q, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Section Divider ──────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── Explainability Panel — 9 tabs ────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-title">🧠 Explainability — SHAP Analysis</div>',
    unsafe_allow_html=True,
)

tab_names = [
    "📊 Importance", "📈 Distribution", "🗺️ Heatmap",
    "🌡️ Temp", "🔄 Cycle", "⚡ Current",
    "🎯 Action", "🏆 Ranking", "🔗 GRU+RL",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    try:
        fig = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")
        print(f"XAI Error in tab 0: {str(e)}")

with tabs[1]:
    try:
        fig = generate_shap_distribution_plot(shap_values, rl_state, chosen_action=chosen_action)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Distribution plot unavailable")
        print(f"XAI Error in tab 1: {str(e)}")

with tabs[2]:
    try:
        fig = generate_shap_heatmap(shap_values)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Heatmap unavailable")
        print(f"XAI Error in tab 2: {str(e)}")

with tabs[3]:
    try:
        fig = generate_temperature_dependence(dqn_model, rl_state)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Temperature dependence plot unavailable")
        print(f"XAI Error in tab 3: {str(e)}")

with tabs[4]:
    try:
        fig = generate_cycle_dependence(dqn_model, rl_state)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Cycle influence plot unavailable")
        print(f"XAI Error in tab 4: {str(e)}")

with tabs[5]:
    try:
        fig = generate_current_dependence(dqn_model, rl_state)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Current influence plot unavailable")
        print(f"XAI Error in tab 5: {str(e)}")

with tabs[6]:
    try:
        fig = generate_action_influence_plot(shap_values)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Action influence plot unavailable")
        print(f"XAI Error in tab 6: {str(e)}")

with tabs[7]:
    try:
        fig = generate_feature_ranking_plot(shap_values)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Feature ranking plot unavailable")
        print(f"XAI Error in tab 7: {str(e)}")

with tabs[8]:
    try:
        fig = generate_combined_xai_plot(shap_values, rl_state, gru_raw_inputs, chosen_action)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Combined GRU + RL plot unavailable")
        print(f"XAI Error in tab 8: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)


# ── AI Reasoning ─────────────────────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">💡 AI Reasoning</div>', unsafe_allow_html=True)

reasoning = generate_reasoning_text(action, cal_soh, st.session_state.tavg, est_cycle, est_current)
st.markdown(f'<div class="explanation-box">{reasoning}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
