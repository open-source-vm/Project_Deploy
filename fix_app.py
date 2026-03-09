with open("app.py", "r") as f:
    code = f.read()

# 1. Update Plotly charts to use use_container_width=True
code = code.replace("st.plotly_chart(fig_g, use_container_width=True)", "st.plotly_chart(fig_g, use_container_width=True)")
code = code.replace("st.plotly_chart(fig_q, use_container_width=True)", "st.plotly_chart(fig_q, use_container_width=True)")

# Update app.py plt render calls
code = code.replace("st.pyplot(fig)", "st.pyplot(fig, use_container_width=True)")
code = code.replace("st.pyplot(fig1)", "st.pyplot(fig1, use_container_width=True)")
code = code.replace("st.pyplot(fig2)", "st.pyplot(fig2, use_container_width=True)")

# Update autosize and margins in app.py Plotly figures
gauge_old = "margin=dict(l=30, r=30, t=20, b=10),"
gauge_new = "margin=dict(l=20, r=20, t=40, b=20), autosize=True,"
if gauge_old in code:
    code = code.replace(gauge_old, gauge_new)

qchart_old = "margin=dict(l=30, r=30, t=40, b=25),"
qchart_new = "margin=dict(l=20, r=20, t=40, b=20), autosize=True,"
if qchart_old in code:
    code = code.replace(qchart_old, qchart_new)

with open("app.py", "w") as f:
    f.write(code)

print("APP UPDATED")
