import sys
import json

with open("app.py", "r") as f:
    lines = f.readlines()

# lines 96-194 (0-indexed: 95-194)
target1 = "".join(lines[95:194])

replacement1 = """# ══════════════════════════════════════════════════════════════════════════════
#  LED TOGGLE SWITCH
# ══════════════════════════════════════════════════════════════════════════════

led_switch = components.declare_component("led_switch", path="assets/led_switch")
# Default to current session state to avoid None on first render causing issues later
led_res = led_switch(checked=st.session_state.show_inputs, key="led_toggle", default=st.session_state.show_inputs)

if led_res is not None and bool(led_res) != st.session_state.show_inputs:
    st.session_state.show_inputs = bool(led_res)
    st.rerun()
"""

target2 = "".join(lines[195:307])

replacement2 = """# ══════════════════════════════════════════════════════════════════════════════
#  BATTERY SENSOR INPUTS — Neumorphic custom component
# ══════════════════════════════════════════════════════════════════════════════

neu_inputs_component = components.declare_component("neu_inputs", path="assets/neu_input")

if st.session_state.show_inputs:
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">🔋 Battery Sensor Inputs</div>',
        unsafe_allow_html=True,
    )

    defaults = {
        "ir": st.session_state.ir,
        "tavg": st.session_state.tavg,
        "qc": st.session_state.qc,
        "tmax": st.session_state.tmax,
        "qd": st.session_state.qd,
        "chargetime": st.session_state.chargetime
    }
    
    input_vals = neu_inputs_component(defaults=defaults, key="neu_battery_inputs", default=defaults)
    
    if input_vals is not None:
        st.session_state.ir = float(input_vals.get("ir", st.session_state.ir))
        st.session_state.tavg = float(input_vals.get("tavg", st.session_state.tavg))
        st.session_state.qc = float(input_vals.get("qc", st.session_state.qc))
        st.session_state.tmax = float(input_vals.get("tmax", st.session_state.tmax))
        st.session_state.qd = float(input_vals.get("qd", st.session_state.qd))
        st.session_state.chargetime = int(float(input_vals.get("chargetime", st.session_state.chargetime)))

    st.markdown('</div>', unsafe_allow_html=True)
"""

content = "".join(lines[:95]) + replacement1 + "".join(lines[194:195]) + replacement2 + "".join(lines[307:])

with open("app.py", "w") as f:
    f.write(content)
print("SUCCESS")
