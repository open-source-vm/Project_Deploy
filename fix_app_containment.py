import re

with open("app.py", "r") as f:
    code = f.read()

# 1. Add global st.markdown padding fix for .block-container
global_padding = """# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    ''',
    unsafe_allow_html=True
)
def _load_css(path):"""
if "# \u2500\u2500 CSS \u2500\u2500" in code and ".block-container" not in code:
    code = code.replace("""# ── CSS ──────────────────────────────────────────────────────────────────────
def _load_css(path):""", global_padding)


# 2. Wrap Sensor Inputs in container
sensor_inputs_old = """if st.session_state.show_inputs:
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title" style="margin-bottom: 20px;">🔧 Battery Sensor Inputs</div>',
        unsafe_allow_html=True,
    )"""

sensor_inputs_new = """if st.session_state.show_inputs:
    with st.container():
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title" style="margin-bottom: 20px;">🔧 Battery Sensor Inputs</div>',
            unsafe_allow_html=True,
        )"""

if sensor_inputs_old in code:
    print("Patching sensor inputs (simple match)...")
    code = code.replace(sensor_inputs_old, sensor_inputs_new)
    # the rest of the block needs indentation, easier to do with regex or manual chunking
    # Let's just do a regex replace for the entire block until the next comment
    
# More reliable way to indent the sensor block:
sensor_block_match = re.search(r'(if st\.session_state\.show_inputs:\n)(.*?)(# ── Load Models)', code, re.DOTALL)
if sensor_block_match and "with st.container():" not in sensor_block_match.group(2):
    inner_code = sensor_block_match.group(2)
    indented_inner = "\n".join("    " + line if line.strip() else line for line in inner_code.split("\n"))
    new_block = sensor_block_match.group(1) + "    with st.container():\n" + indented_inner + "# ── Load Models"
    code = code[:sensor_block_match.start()] + new_block + code[sensor_block_match.end():]
    
# 3. Wrap Prediction Panels 
pred_block_old = """# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT  — Row 1: Battery Health | Charging Decision
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:"""

pred_block_new = """# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT  — Row 1: Battery Health | Charging Decision
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:"""
if pred_block_old in code:
    code = code.replace(pred_block_old, pred_block_new)
    # We need to indent everything under col1 and col2 up to Explainability
    pred_match = re.search(r'(    with col1:\n)(.*?)(# ── Explainability)', code, re.DOTALL)
    if pred_match:
        inner = pred_match.group(2)
        indented = "\n".join("    " + line if line.strip() else line for line in inner.split("\n"))
        code = code[:pred_match.start()] + pred_match.group(1) + indented + "\n# ── Explainability" + code[pred_match.end():]


# 4. Wrap Explainability
exp_old = """# ── Explainability Panel ────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)"""

exp_new = """st.divider()
# ── Explainability Panel ────────────────────────────────────────────
with st.container():
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)"""
if exp_old in code:
    code = code.replace(exp_old, exp_new)
    exp_match = re.search(r'(    st\.markdown\(\'<div class="neu-card">\', unsafe_allow_html=True\)\n)(.*?)(# ── AI Reasoning)', code, re.DOTALL)
    if exp_match:
        inner = exp_match.group(2)
        indented = "\n".join("    " + line if line.strip() else line for line in inner.split("\n"))
        code = code[:exp_match.start()] + exp_match.group(1) + indented + "\n# ── AI Reasoning" + code[exp_match.end():]

# 5. Wrap AI Reasoning
rsn_old = """# ── AI Reasoning ─────────────────────────────────────────────────────────────
st.markdown('<div class="neu-card">', unsafe_allow_html=True)"""

rsn_new = """st.divider()
# ── AI Reasoning ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)"""
if rsn_old in code:
    code = code.replace(rsn_old, rsn_new)
    rsn_match = re.search(r'(    st\.markdown\(\'<div class="neu-card">\', unsafe_allow_html=True\)\n)(.*?)(# ── Footer)', code, re.DOTALL)
    if rsn_match:
        inner = rsn_match.group(2)
        indented = "\n".join("    " + line if line.strip() else line for line in inner.split("\n"))
        code = code[:rsn_match.start()] + rsn_match.group(1) + indented + "\n# ── Footer" + code[rsn_match.end():]

# 6. Apply Chart Dimensions (Plotly heights and matplotlib fig sizes)
if 'st.plotly_chart(fig_g, use_container_width=True)' in code:
    code = code.replace('st.plotly_chart(fig_g, use_container_width=True)', 'st.plotly_chart(fig_g, use_container_width=True, height=350)')
if 'st.plotly_chart(fig_q, use_container_width=True)' in code:
    code = code.replace('st.plotly_chart(fig_q, use_container_width=True)', 'st.plotly_chart(fig_q, use_container_width=True, height=350)')

with open("app.py", "w") as f:
    f.write(code)

print("CONTAINERS APPLIED")
