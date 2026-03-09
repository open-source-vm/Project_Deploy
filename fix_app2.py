import re

with open("app.py", "r") as f:
    code = f.read()

# Make sure all subheaders are used inside columns for charts
if "st.markdown(\"<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>SoH Prediction Feature Importance (GRU)</p>\", unsafe_allow_html=True)" in code:
    code = code.replace(
        "st.markdown(\"<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>SoH Prediction Feature Importance (GRU)</p>\", unsafe_allow_html=True)",
        "st.subheader('SoH Prediction Feature Importance (GRU)')"
    )

if "st.markdown(\"<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>RL Decision Feature Influence</p>\", unsafe_allow_html=True)" in code:
    code = code.replace(
        "st.markdown(\"<p style='text-align:center; font-weight:600; font-size:14px; color:#64748b;'>RL Decision Feature Influence</p>\", unsafe_allow_html=True)",
        "st.subheader('RL Decision Feature Influence')"
    )

with open("app.py", "w") as f:
    f.write(code)
print("APP ALIGNED")
