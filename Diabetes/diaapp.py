import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🏥",
    layout="wide"
)

# Simple CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        text-align: center;
    }
    
    .prediction-result {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid #007bff;
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/xgboost.pkl', 'rb') as file:
            return joblib.load(file)
    except FileNotFoundError:
        st.warning("Model file not found. Using demo mode.")
        return None

model = load_model()

# Generate sample data
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'Glucose': np.random.normal(120, 30, n),
        'BloodPressure': np.random.normal(80, 15, n),
        'BMI': np.random.normal(25, 5, n),
        'Age': np.random.randint(20, 80, n),
        'Pregnancies': np.random.randint(0, 8, n),
        'SkinThickness': np.random.normal(25, 8, n),
        'Insulin': np.random.normal(100, 40, n),
        'DiabetesPedigree': np.random.uniform(0.1, 2.0, n),
        'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35])
    })
    return data

sample_data = get_sample_data()

# Title
st.title('🏥 Diabetes Predictor App')
st.markdown("---")

# SECTION 1: Input Parameters
st.header("1. Enter Patient Information")
col1, col2 = st.columns(2)

with col1:
    glucose = st.slider('Glucose Level', 0, 200, 120)
    blood_pressure = st.slider('Blood Pressure', 0, 150, 80)
    bmi = st.slider('BMI', 0.0, 50.0, 25.0, 0.1)
    age = st.slider('Age', 0, 120, 35)

with col2:
    pregnancies = st.slider('Pregnancies', 0, 20, 1)
    skin_thickness = st.slider('Skin Thickness', 0, 100, 25)
    insulin = st.slider('Insulin Level', 0, 300, 100)
    diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, 0.01)

st.markdown("---")

# SECTION 2: Current Values Display
st.header("2. Current Input Values")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h4>Glucose</h4>
        <h3>{glucose}</h3>
        <small>mg/dL</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-box">
        <h4>BMI</h4>
        <h3>{bmi}</h3>
        <small>kg/m²</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h4>Blood Pressure</h4>
        <h3>{blood_pressure}</h3>
        <small>mmHg</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-box">
        <h4>Age</h4>
        <h3>{age}</h3>
        <small>years</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h4>Pregnancies</h4>
        <h3>{pregnancies}</h3>
        <small>count</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-box">
        <h4>Skin Thickness</h4>
        <h3>{skin_thickness}</h3>
        <small>mm</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <h4>Insulin</h4>
        <h3>{insulin}</h3>
        <small>μU/mL</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-box">
        <h4>Pedigree</h4>
        <h3>{diabetes_pedigree:.2f}</h3>
        <small>function</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# SECTION 3: Prediction
st.header("3. Risk Prediction")

if st.button('Predict Diabetes Risk', type='primary'):
    if model is not None:
        input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, 
                               bmi, diabetes_pedigree, age, pregnancies]])
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-result" style="border-color: #dc3545;">
                    <h2 style="color: #dc3545;">HIGH RISK</h2>
                    <h4>Risk Probability: {prediction_proba[1]:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result" style="border-color: #28a745;">
                    <h2 style="color: #28a745;">LOW RISK</h2>
                    <h4>Risk Probability: {prediction_proba[0]:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        # Demo prediction
        demo_risk = np.random.choice([0, 1], p=[0.7, 0.3])
        demo_prob = np.random.uniform(0.6, 0.9)
        
        if demo_risk == 1:
            st.markdown(f"""
            <div class="prediction-result" style="border-color: #dc3545;">
                <h2 style="color: #dc3545;">HIGH RISK (Demo)</h2>
                <h4>Risk Probability: {demo_prob:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result" style="border-color: #28a745;">
                <h2 style="color: #28a745;">LOW RISK (Demo)</h2>
                <h4>Risk Probability: {demo_prob:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# SECTION 4: Simple Charts
st.header("4. Data Visualization")

# Chart 1: Bar chart of input values
st.subheader("4.1 Your Input Values")
input_values = pd.DataFrame({
    'Parameter': ['Glucose', 'Blood Pressure', 'BMI', 'Age', 'Pregnancies', 
                 'Skin Thickness', 'Insulin', 'Pedigree*100'],
    'Value': [glucose, blood_pressure, bmi, age, pregnancies, 
             skin_thickness, insulin, diabetes_pedigree*100]
})

fig1 = px.bar(input_values, x='Parameter', y='Value', 
              title='Your Health Parameters',
              color_discrete_sequence=['#007bff'])
fig1.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color='black'
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Histogram comparison
st.subheader("4.2 Population Distribution Comparison")
col1, col2 = st.columns(2)

with col1:
    fig2 = px.histogram(sample_data, x='Glucose', nbins=20, 
                       title='Glucose Level Distribution in Population',
                       color_discrete_sequence=['#6c757d'])
    fig2.add_vline(x=glucose, line_dash="dash", line_color="red", 
                   annotation_text="Your Value")
    fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.histogram(sample_data, x='BMI', nbins=20,
                       title='BMI Distribution in Population',
                       color_discrete_sequence=['#6c757d'])
    fig3.add_vline(x=bmi, line_dash="dash", line_color="red", 
                   annotation_text="Your Value")
    fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
    st.plotly_chart(fig3, use_container_width=True)

# Chart 3: Scatter plot
st.subheader("4.3 Age vs Health Metrics")
fig4 = px.scatter(sample_data, x='Age', y='Glucose', 
                  title='Age vs Glucose Level in Population',
                  color_discrete_sequence=['#6c757d'])
fig4.add_scatter(x=[age], y=[glucose], mode='markers', 
                marker=dict(color='red', size=15), name='Your Values')
fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
st.plotly_chart(fig4, use_container_width=True)

# Chart 4: Simple correlation heatmap
st.subheader("4.4 Feature Relationships")
correlation_data = sample_data[['Glucose', 'BloodPressure', 'BMI', 'Age']].corr()

fig5 = px.imshow(correlation_data, 
                title='Correlation Between Key Health Metrics',
                color_continuous_scale='RdBu_r',
                aspect='auto')
fig5.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
st.plotly_chart(fig5, use_container_width=True)

# Chart 5: Risk by age groups (simple bar chart)
st.subheader("4.5 Diabetes Risk by Age Group")
sample_data['AgeGroup'] = pd.cut(sample_data['Age'], 
                                bins=[0, 30, 45, 60, 100], 
                                labels=['Under 30', '30-45', '45-60', 'Over 60'])
age_risk = sample_data.groupby('AgeGroup')['Outcome'].mean().reset_index()
age_risk['Risk_Percentage'] = age_risk['Outcome'] * 100

fig6 = px.bar(age_risk, x='AgeGroup', y='Risk_Percentage',
              title='Average Diabetes Risk by Age Group (%)',
              color_discrete_sequence=['#007bff'])
fig6.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# SECTION 5: Summary
st.header("5. Summary & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Health Status Summary")
    
    # Simple status indicators
    if glucose > 140:
        st.error("🔴 Glucose: High")
    elif glucose > 100:
        st.warning("🟡 Glucose: Elevated")
    else:
        st.success("🟢 Glucose: Normal")
        
    if blood_pressure > 120:
        st.error("🔴 Blood Pressure: High")
    elif blood_pressure > 80:
        st.warning("🟡 Blood Pressure: Elevated")
    else:
        st.success("🟢 Blood Pressure: Normal")
        
    if bmi > 30:
        st.error("🔴 BMI: Obese")
    elif bmi > 25:
        st.warning("🟡 BMI: Overweight")
    else:
        st.success("🟢 BMI: Normal")

with col2:
    st.subheader("General Recommendations")
    st.info("💊 Consult healthcare professionals for medical advice")
    st.info("🥗 Maintain a balanced, low-sugar diet")
    st.info("🏃‍♂️ Regular physical exercise (30 min/day)")
    st.info("📊 Monitor blood glucose regularly")
    st.info("💧 Stay hydrated and maintain healthy weight")

st.markdown("---")
st.markdown("**Note:** This prediction is based on machine learning models and should not replace professional medical advice.")
st.markdown("*Made with Streamlit*")