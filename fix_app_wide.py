import re

with open("app.py", "r") as f:
    code = f.read()

# 1. Update st.set_page_config layout="wide"
# It currently has layout="wide" already, but let's make sure it's strictly correct
# st.set_page_config(
#     page_title="Explainable AI EV-BMS Dashboard",
#     page_icon="⚡",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )
code = code.replace('page_title="Explainable AI EV-BMS Dashboard"', 'page_title="Explainable AI EV Battery Management Dashboard"')
code = re.sub(r'st\.set_page_config\([^)]+\)', '''st.set_page_config(
    page_title="Explainable AI EV Battery Management Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)''', code)

# 2. Make Header compact
header_old = """st.markdown(
    '<h1 class="dashboard-title">'
    'Explainable AI EV Battery Management Dashboard'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="dashboard-subtitle">'
    'AI-driven battery health prediction and charging optimisation &nbsp;·&nbsp; '
    'GRU + Double DQN + SHAP'
    '</p>',
    unsafe_allow_html=True,
)"""

header_new = """st.markdown(
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
)"""
if header_old in code:
    code = code.replace(header_old, header_new)

# 3. Explainability side-by-side single card
# Already did this somewhat, but let's make sure the bounding box card contains both
explainability_old = """# ── Explainability Panel — 9 tabs ────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-title">📊 Explainability — SHAP Analysis</div>'
    '<div class="card-subtitle">Feature impact on AI model predictions</div>',
    unsafe_allow_html=True,
)

col_ex1, col_ex2 = st.columns(2)
with col_ex1:"""

explainability_new = """# ── Explainability Panel ────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="card-title">📊 Model Explainability (SHAP)</div>'
    '<div class="card-subtitle" style="font-size: 0.85em; color: #64748b; margin-bottom: 15px;">Feature impact on AI model predictions</div>',
    unsafe_allow_html=True,
)

col_ex1, col_ex2 = st.columns(2)
with col_ex1:"""
if explainability_old in code:
    code = code.replace(explainability_old, explainability_new)


# 4. Remove Section Divider to avoid whitespace bloat
code = code.replace("""# ── Section Divider ──────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)""", "")

with open("app.py", "w") as f:
    f.write(code)

print("APP WIDE UPDATED")
