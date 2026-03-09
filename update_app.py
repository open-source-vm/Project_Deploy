import re

with open("app.py", "r") as f:
    code = f.read()

# 1. Update Header
code = code.replace(
    "'⚡ Explainable AI EV Battery Management Dashboard'",
    "'Explainable AI EV Battery Management Dashboard'"
)

# 2. Update Icons in headers inside app.py
# Battery Health
code = code.replace(
    '<div class="card-title">🔋 Battery Health</div>',
    '<div class="card-title">🔋 Battery Health</div>'
)
# Charging Decision
code = code.replace(
    '<div class="card-title">⚙️ Charging Decision</div>',
    '<div class="card-title">⚡ Charging Decision (RL Policy)</div>'
)
# Explainability
code = code.replace(
    '<div class="card-title">🧠 Explainability — SHAP Analysis</div>',
    '<div class="card-title">📊 Explainability — SHAP Analysis</div>\n    <div class="card-subtitle">Feature impact on AI model predictions</div>'
)
# AI Reasoning
code = code.replace(
    '<div class="card-title">💡 AI Reasoning</div>',
    '<div class="card-title">🤖 AI Reasoning</div>'
)

# 3. Update Battery Health badge logic
# We need to replace the cal_pct logic where it sets the badge.
# Find: badge_key = health_label.lower().replace(" ", "-") ... st.markdown(...)
health_badge_old = """    # Health badge
    badge_key = health_label.lower().replace(" ", "-")
    st.markdown(
        f'<div style="text-align:center;margin:0.5rem 0">'
        f'<span class="health-badge health-{badge_key}">{health_label}</span></div>',
        unsafe_allow_html=True,
    )"""

health_badge_new = """    # Health badge
    if cal_pct > 90:
        badge_color = "#2ECC71"
    elif cal_pct >= 70:
        badge_color = "#F39C12"
    else:
        badge_color = "#E74C3C"

    st.markdown(
        f'<div style="text-align:center;margin:0.5rem 0">'
        f'<span class="status-badge" style="background-color: {badge_color}; color: white;">Status: {health_label}</span></div>',
        unsafe_allow_html=True,
    )"""
if health_badge_old in code:
    code = code.replace(health_badge_old, health_badge_new)
else:
    print("WARNING: health_badge_old not found!")

# 4. Charging Decision Text
# Find: st.markdown(f'<div class="action-badge {action_class}">{action_text}</div>', ...
action_badge_old = """    st.markdown(
        f'<div class="action-badge {action_class}">{action_text}</div>',
        unsafe_allow_html=True,
    )"""
action_badge_new = """    st.markdown(
        f'<div style="text-align:center; font-weight:600; margin-bottom: 8px; color: #4A90E2;">Selected Action: {action_text.replace("⬇  Decrease Charging", "Decrease Charging").replace("⏸  Maintain Charging", "Maintain Charging").replace("⬆  Increase Charging", "Increase Charging")}</div>',
        unsafe_allow_html=True,
    )"""
if action_badge_old in code:
    code = code.replace(action_badge_old, action_badge_new)
else:
    print("WARNING: action_badge_old not found!")

action_text_map_old = """action_text_map = {
    0: "⬇  Decrease Charging",
    1: "⏸  Maintain Charging",
    2: "⬆  Increase Charging",
}"""
action_text_map_new = """action_text_map = {
    0: "Decrease Charging",
    1: "Maintain Charging",
    2: "Increase Charging",
}"""
code = code.replace(action_text_map_old, action_text_map_new)

# 5. Highlight selected action in bar chart
bar_chart_old = """    fig_q = go.Figure(data=[go.Bar(
        x=["Decrease", "Maintain", "Increase"],
        y=q_values.tolist(),
        marker_color=["#ef4444", "#f59e0b", "#22c55e"],
        marker_line=dict(color="#c5cbd3", width=1),
        text=[f"{v:.4f}" for v in q_values],
        textposition="outside",
        textfont=dict(color="#333333", size=11),
    )])"""

bar_chart_new = """    colors = ["#E74C3C", "#F39C12", "#2ECC71"]
    opacities = [0.4, 0.4, 0.4]
    line_widths = [0, 0, 0]
    opacities[action] = 1.0
    line_widths[action] = 2
    
    fig_q = go.Figure(data=[go.Bar(
        x=["Decrease", "Maintain", "Increase"],
        y=q_values.tolist(),
        marker_color=colors,
        opacity=opacities,
        marker_line=dict(color="#333", width=line_widths),
        text=[f"{v:.4f}" for v in q_values],
        textposition="outside",
        textfont=dict(color="#333333", size=11),
    )])"""
if bar_chart_old in code:
    code = code.replace(bar_chart_old, bar_chart_new)
else:
    print("WARNING: bar_chart_old not found!")

# 6. Replace Tabs with 2 columns in Explainability
tabs_old_pattern = re.compile(r"tab_names = \[\n.*?st\.markdown\(\"</div>\", unsafe_allow_html\=True\)", re.DOTALL)
tabs_replacement = """col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.markdown("<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>SoH Prediction Feature Importance (GRU)</p>", unsafe_allow_html=True)
    try:
        fig1 = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
        st.pyplot(fig1)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")

with col_ex2:
    st.markdown("<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>RL Decision Feature Influence</p>", unsafe_allow_html=True)
    try:
        fig2 = generate_action_influence_plot(shap_values)
        st.pyplot(fig2)
    except Exception as e:
        st.warning("Plot unavailable")

st.markdown("</div>", unsafe_allow_html=True)"""

code = tabs_old_pattern.sub(tabs_replacement, code)

# 7. AI Reasoning structure
ai_reasoning_old = """reasoning = generate_reasoning_text(action, cal_soh, st.session_state.tavg, est_cycle, est_current)
st.markdown(f'<div class="explanation-box">{reasoning}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)"""

ai_reasoning_new = """reasoning = generate_reasoning_text(action, cal_soh, st.session_state.tavg, est_cycle, est_current)
st.markdown(
f'''
<div class="reasoning-card">
    <div style="font-weight: bold; margin-bottom: 8px; color: #4A90E2;">Decision: {action_text}</div>
    {reasoning}
</div>
''', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)"""
if "reasoning = generate_reasoning_text" in code:
    code = code.replace(ai_reasoning_old, ai_reasoning_new)
else:
    print("WARNING: ai_reasoning_old not found!")

# 8. Add Footer
if "Explainable AI powered by SHAP" not in code:
    code += """
# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('''
<div class="footer">
    EV Battery AI System<br>
    GRU SoH Prediction + Double DQN Charging Optimization<br>
    Explainable AI powered by SHAP
</div>
''', unsafe_allow_html=True)
"""

with open("app.py", "w") as f:
    f.write(code)

print("APP.PY UPDATED")
