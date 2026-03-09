import streamlit as st
import streamlit.components.v1 as components

value = components.html("""
<button onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'hello'}, '*')">Click</button>
""")
st.write(f"Value: {value}")
