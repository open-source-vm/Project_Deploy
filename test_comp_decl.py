import streamlit as st
import streamlit.components.v1 as components
import os

os.makedirs("test_comp_dir", exist_ok=True)
with open("test_comp_dir/index.html", "w") as f:
    f.write("""
    <button onclick="
      var val = new Date().getTime();
      window.parent.postMessage({
        isStreamlitMessage: true,
        type: 'streamlit:setComponentValue',
        value: val
      }, '*');
    ">Click me</button>
    """)

comp = components.declare_component("my_comp", path="test_comp_dir")
val = comp(key="test1")
print(f"VAL={val}")
st.write(f"VAL={val}")
