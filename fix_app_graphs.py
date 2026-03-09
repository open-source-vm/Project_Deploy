import re

with open("app.py", "r") as f:
    code = f.read()

graphs_old = """col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.subheader('SoH Prediction Feature Importance (GRU)')
    try:
        fig1 = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
        st.pyplot(fig1, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")

with col_ex2:
    st.subheader('RL Decision Feature Influence')
    try:
        fig2 = generate_action_influence_plot(shap_values)
        st.pyplot(fig2, use_container_width=True)
    except Exception as e:
        st.warning("Plot unavailable")"""

graphs_new = """row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.subheader('Global Feature Importance (GRU)')
    try:
        fig1 = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
        st.pyplot(fig1, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")

with row1_col2:
    st.subheader('Feature Distribution')
    try:
        fig2 = generate_shap_distribution_plot(shap_values, rl_state, chosen_action=chosen_action)
        st.pyplot(fig2, use_container_width=True)
    except Exception as e:
        st.warning("Distribution plot unavailable")

with row1_col3:
    st.subheader('SHAP Heatmap')
    try:
        fig3 = generate_shap_heatmap(shap_values)
        st.pyplot(fig3, use_container_width=True)
    except Exception as e:
        st.warning("Heatmap unavailable")

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.subheader('Temperature Influence')
    try:
        fig4 = generate_temperature_dependence(dqn_model, rl_state)
        st.pyplot(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")

with row2_col2:
    st.subheader('Cycle Influence')
    try:
        fig5 = generate_cycle_dependence(dqn_model, rl_state)
        st.pyplot(fig5, use_container_width=True)
    except Exception as e:
        st.warning("Plot unavailable")

with row2_col3:
    st.subheader('Current Influence')
    try:
        fig6 = generate_current_dependence(dqn_model, rl_state)
        st.pyplot(fig6, use_container_width=True)
    except Exception as e:
        st.warning("Plot unavailable")
        
row3_col1, row3_col2, row3_col3 = st.columns(3)

with row3_col1:
    st.subheader('RL Action Influence')
    try:
        fig7 = generate_action_influence_plot(shap_values)
        st.pyplot(fig7, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot unavailable: {str(e)}")

with row3_col2:
    st.subheader('RL Decision SHAP Summary')
    try:
        fig8 = generate_feature_ranking_plot(shap_values)
        st.pyplot(fig8, use_container_width=True)
    except Exception as e:
        st.warning("Plot unavailable")

with row3_col3:
    st.subheader('GRU vs RL Feature Ranking')
    try:
        fig9 = generate_combined_xai_plot(shap_values, rl_state, gru_raw_inputs, chosen_action)
        st.pyplot(fig9, use_container_width=True)
    except Exception as e:
        st.warning("Plot unavailable")"""

if graphs_old in code:
    code = code.replace(graphs_old, graphs_new)
    print("Graphs replaced")

with open("app.py", "w") as f:
    f.write(code)

