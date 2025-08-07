import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px

# =========================
# Custom CSS for background (Blue-Teal gradient)
# =========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #e0f7fa 0%, #80deea 100%);
}
[data-testid="stSidebar"] {
    background-color: #004d61;
    color: white;
}
h1, h2, h3, h4 {
    color: #00363a;
    font-family: 'Segoe UI', sans-serif;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# Load Model
# =========================
model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost.pkl')
with open(model_path, 'rb') as file:
    model = joblib.load(file)

# =========================
# Page Config
# =========================
st.set_page_config(page_title="🩺 Diabetes Predictor", layout="wide")
st.title("🩺 Diabetes Predictor Dashboard")
st.subheader("Visualize and Predict Diabetes Risk from Your Health Metrics")
st.markdown("---")

# =========================
# Sidebar - Inputs
# =========================
st.sidebar.header("📋 Enter Your Details")
pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 1)
glucose = st.sidebar.slider('Glucose Level', 0, 200, 100)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 150, 70)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
insulin = st.sidebar.slider('Insulin Level', 0, 300, 80)
bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0, step=0.1)
diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01)
age = st.sidebar.slider('Age', 0, 120, 30)

# Prepare Input Data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# =========================
# Predict & Sequential Display
# =========================
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # ===== 1. Prediction Result =====
    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes ({probability:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({probability:.2f}%)")

    st.info("Prediction based on an XGBoost model trained on health metrics.")
    st.markdown("---")

    # ===== 2. Pie Chart =====
    st.subheader("📊 Risk Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        [probability, 100 - probability],
        labels=["High Risk", "Low Risk"],
        autopct='%1.1f%%',
        colors=['#ff6b6b', '#6bcf6b'],
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)
    st.markdown("---")

    # ===== 3. Probability Bar Chart =====
    st.subheader("📈 Prediction Probability")
    prob_df = {
        "Risk Type": ["Low Risk", "High Risk"],
        "Probability (%)": [100 - probability, probability]
    }
    fig_bar = px.bar(
        prob_df,
        x="Risk Type",
        y="Probability (%)",
        color="Risk Type",
        color_discrete_map={"Low Risk": "#6bcf6b", "High Risk": "#ff6b6b"},
        text="Probability (%)"
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("---")

    # ===== 4. Radar Chart =====
    st.subheader("📌 Your Health Metrics Overview")
    metrics = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "DPF", "Age"]
    values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

    fig_radar = px.line_polar(
        r=values,
        theta=metrics,
        line_close=True,
        markers=True
    )
    fig_radar.update_traces(fill='toself', line_color='#1f77b4')
    fig_radar.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_radar, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Professional Dashboard UI")
