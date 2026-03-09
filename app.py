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
    page_title="Explainable AI EV Battery Management Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style native st.container(border=True) to look like our cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08) !important;
        border: none !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12) !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)
def _load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
try:
    _load_css(os.path.join("assets", "style.css"))
except FileNotFoundError:
    pass

# ── Custom Streamlit Components ──────────────────────────────────────────────
led_switch = components.declare_component("led_switch", path="assets/led_switch")
neu_inputs_component = components.declare_component("neu_inputs", path="assets/neu_input")

# ── Initialize Session State ─────────────────────────────────────────────────
defaults = {
    "ir": 0.04,
    "qc": 1.5,
    "qd": 1.5,
    "tavg": 30.0,
    "tmax": 35.0,
    "chargetime": 5000,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = True

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER — title + subtitle
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '''
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 class="dashboard-title" style="margin-bottom: 5px;">⚡ Explainable AI EV Battery Management Dashboard</h1>
        <p class="dashboard-subtitle" style="margin-top: 0; font-size: 0.9em; font-weight: 300; color: #64748b;">
            AI-driven battery health prediction and charging optimisation<br>
            GRU + Double DQN + SHAP
        </p>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── LED Toggle Switch (custom component) ─────────────────────────────────────
led_res = led_switch(
    checked=st.session_state.show_inputs,
    key="led_toggle",
    default=st.session_state.show_inputs,
)

if led_res is not None and bool(led_res) != st.session_state.show_inputs:
    st.session_state.show_inputs = bool(led_res)
    st.rerun()

# ── Battery Sensor Inputs panel (collapsible in main page) ───────────────────
if st.session_state.show_inputs:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title" style="margin-bottom: 20px;">🔧 Battery Sensor Inputs</div>',
            unsafe_allow_html=True,
        )
        input_defaults = {
            "ir": st.session_state.ir,
            "tavg": st.session_state.tavg,
            "qc": st.session_state.qc,
            "tmax": st.session_state.tmax,
            "qd": st.session_state.qd,
            "chargetime": st.session_state.chargetime,
        }
        input_vals = neu_inputs_component(
            defaults=input_defaults, key="neu_battery_inputs", default=input_defaults
        )
        if input_vals is not None:
            st.session_state.ir = float(input_vals.get("ir", st.session_state.ir))
            st.session_state.tavg = float(input_vals.get("tavg", st.session_state.tavg))
            st.session_state.qc = float(input_vals.get("qc", st.session_state.qc))
            st.session_state.tmax = float(input_vals.get("tmax", st.session_state.tmax))
            st.session_state.qd = float(input_vals.get("qd", st.session_state.qd))
            st.session_state.chargetime = int(
                float(input_vals.get("chargetime", st.session_state.chargetime))
            )

# ── Load Models ──────────────────────────────────────────────────────────────
with st.spinner("Loading ML models …"):
    gru_model, dqn_model, scaler_X, scaler_y = load_models()

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE (no UI — pure computation)
# ══════════════════════════════════════════════════════════════════════════════
sequence = prepare_gru_sequence(
    scaler_X,
    st.session_state.ir, st.session_state.qc, st.session_state.qd,
    st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime,
)
raw_soh = predict_soh(gru_model, scaler_y, sequence)

est_cycle = estimate_cycle(st.session_state.qd, st.session_state.qc)
est_current = estimate_current(st.session_state.qc, st.session_state.chargetime)

cal_soh = calibrate_soh(raw_soh, est_cycle)
health_label = get_battery_health_label(cal_soh)

rl_state = construct_rl_state(cal_soh, st.session_state.tavg, est_cycle, est_current)
action, q_values = predict_rl_action(dqn_model, rl_state)

action_map = {0: "decrease", 1: "maintain", 2: "increase"}
action_class = action_map.get(action, "maintain")
action_text_map = {
    0: "Decrease Charging",
    1: "Maintain Charging",
    2: "Increase Charging",
}
action_text = action_text_map.get(action, "Maintain Charging")

shap_values, chosen_action, _ = compute_shap_values(dqn_model, rl_state)
gru_raw_inputs = [
    st.session_state.ir, st.session_state.qc, st.session_state.qd,
    st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime,
]

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION PANELS — Battery Health | Charging Decision  (side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
col_h, col_a = st.columns(2)

with col_h:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title" style="margin-bottom: 20px;">🔋 Battery Health</div>',
            unsafe_allow_html=True,
        )

        cal_pct = cal_soh * 100
        raw_pct = raw_soh * 100

        st.markdown(f'<div class="soh-value">{cal_pct:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="soh-label">Calibrated State-of-Health</div>', unsafe_allow_html=True
        )

        if cal_pct > 90:
            badge_color = "#2ECC71"
        elif cal_pct >= 70:
            badge_color = "#F39C12"
        else:
            badge_color = "#E74C3C"

        st.markdown(
            f'<div style="text-align:center;margin:0.5rem 0">'
            f'<span class="status-badge" style="background-color: {badge_color}; color: white;">'
            f'Status: {health_label}</span></div>',
            unsafe_allow_html=True,
        )

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
                "axis": {
                    "range": [0, 100], "tickcolor": "#94a3b8", "tickwidth": 1,
                    "tickfont": {"color": "#64748b"},
                },
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#e2e8f0",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 70], "color": "rgba(239,68,68,0.08)"},
                    {"range": [70, 80], "color": "rgba(251,146,60,0.08)"},
                    {"range": [80, 90], "color": "rgba(245,158,11,0.08)"},
                    {"range": [90, 100], "color": "rgba(34,197,94,0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 2},
                    "thickness": 0.8,
                    "value": cal_pct,
                },
            },
        ))
        fig_g.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333"),
        )
        st.plotly_chart(fig_g, use_container_width=True)

with col_a:
    with st.container(border=True):
        st.markdown(
            '<div class="card-title" style="margin-bottom: 20px;">⚡ Charging Decision (RL Policy)</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="text-align:center; font-weight:600; margin-bottom: 8px; color: #4A90E2;">'
            f'Selected Action: {action_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rl-pills">'
            f'<div class="rl-pill">SoH: <strong>{cal_soh:.4f}</strong></div>'
            f'<div class="rl-pill">Temp: <strong>{st.session_state.tavg:.1f} °C</strong></div>'
            f'<div class="rl-pill">Cycle: <strong>{est_cycle}</strong></div>'
            f'<div class="rl-pill">Current: <strong>{est_current:.2f} A</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        colors = ["#E74C3C", "#F39C12", "#2ECC71"]
        opacities = [0.4, 0.4, 0.4]
        line_widths = [0, 0, 0]
        opacities[action] = 1.0
        line_widths[action] = 2

        fig_q = go.Figure(data=[go.Bar(
            x=["Decrease", "Maintain", "Increase"],
            y=q_values.tolist(),
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(color="#333", width=line_widths),
            ),
            text=[f"{v:.4f}" for v in q_values],
            textposition="outside",
            textfont=dict(color="#333333", size=11),
        )])
        fig_q.update_layout(
            title=dict(text="Q-Values per Action", font=dict(color="#64748b", size=12)),
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333"),
            yaxis=dict(gridcolor="#d1d5db", zerolinecolor="#d1d5db"),
            xaxis=dict(gridcolor="#d1d5db"),
            bargap=0.35,
        )
        st.plotly_chart(fig_q, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  EXPLAINABILITY — 3×3 SHAP Grid
# ══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown(
        '<div class="card-title">📊 Model Explainability (SHAP)</div>'
        '<div style="font-size:0.85em;color:#64748b;margin-bottom:15px;">'
        'Feature impact on AI model predictions</div>',
        unsafe_allow_html=True,
    )

    # Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.subheader("Global Feature Importance (GRU)")
        try:
            fig1 = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
            st.pyplot(fig1, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot unavailable: {e}")
    with r1c2:
        st.subheader("Feature Distribution")
        try:
            fig2 = generate_shap_distribution_plot(shap_values, rl_state, chosen_action=chosen_action)
            st.pyplot(fig2, use_container_width=True)
        except Exception as e:
            st.warning("Distribution plot unavailable")
    with r1c3:
        st.subheader("SHAP Heatmap")
        try:
            fig3 = generate_shap_heatmap(shap_values)
            st.pyplot(fig3, use_container_width=True)
        except Exception as e:
            st.warning("Heatmap unavailable")



    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.subheader("Temperature Influence")
        try:
            fig4 = generate_temperature_dependence(dqn_model, rl_state)
            st.pyplot(fig4, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot unavailable: {e}")
    with r2c2:
        st.subheader("Cycle Influence")
        try:
            fig5 = generate_cycle_dependence(dqn_model, rl_state)
            st.pyplot(fig5, use_container_width=True)
        except Exception as e:
            st.warning("Plot unavailable")
    with r2c3:
        st.subheader("Current Influence")
        try:
            fig6 = generate_current_dependence(dqn_model, rl_state)
            st.pyplot(fig6, use_container_width=True)
        except Exception as e:
            st.warning("Plot unavailable")


    # Row 3
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        st.subheader("RL Action Influence")
        try:
            fig7 = generate_action_influence_plot(shap_values)
            st.pyplot(fig7, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot unavailable: {e}")
    with r3c2:
        st.subheader("RL Decision SHAP Summary")
        try:
            fig8 = generate_feature_ranking_plot(shap_values)
            st.pyplot(fig8, use_container_width=True)
        except Exception as e:
            st.warning("Plot unavailable")
    with r3c3:
        st.subheader("GRU vs RL Feature Ranking")
        try:
            fig9 = generate_combined_xai_plot(shap_values, rl_state, gru_raw_inputs, chosen_action)
            st.pyplot(fig9, use_container_width=True)
        except Exception as e:
            st.warning("Plot unavailable")

# ══════════════════════════════════════════════════════════════════════════════
#  AI REASONING
# ══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown(
        '<div class="card-title" style="margin-bottom: 20px;">🤖 AI Reasoning</div>',
        unsafe_allow_html=True,
    )

    reasoning = generate_reasoning_text(
        action, cal_soh, st.session_state.tavg, est_cycle, est_current
    )
    st.markdown(
        f'''
        <div class="reasoning-card">
            <div style="font-weight: bold; margin-bottom: 8px; color: #4A90E2;">
                Decision: {action_text}
            </div>
            {reasoning}
        </div>
        ''',
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '''
    <div class="footer">
        EV Battery AI System<br>
        GRU SoH Prediction + Double DQN Charging Optimization<br>
        Explainable AI powered by SHAP
    </div>
    ''',
    unsafe_allow_html=True,
)
