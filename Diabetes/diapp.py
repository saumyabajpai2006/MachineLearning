import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="🩺 Diabetes Predictor", layout="wide")

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;}
    .stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .css-1d391kg {background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem;}
    .stSlider > div > div {background: rgba(255,255,255,0.2); border-radius: 15px;}
    .stButton > button {background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border: none; 
                        border-radius: 25px; padding: 0.5rem 2rem; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.3);}
    .metric-box {background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 15px; 
                text-align: center; margin: 1rem 0; backdrop-filter: blur(5px); border: 1px solid rgba(255,255,255,0.2);}
    .high-risk {background: linear-gradient(45deg, #FF6B6B, #FF8E53); padding: 2rem; border-radius: 20px; 
                text-align: center; font-size: 1.5rem; font-weight: bold; animation: pulse 2s infinite;}
    .low-risk {background: linear-gradient(45deg, #4ECDC4, #44A08D); padding: 2rem; border-radius: 20px; 
            text-align: center; font-size: 1.5rem; font-weight: bold; animation: pulse 2s infinite;}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);}}
    .stSidebar {background: rgba(255,255,255,0.1); backdrop-filter: blur(15px);}
    h1, h2, h3 {color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/xgboost.pkl')
    except:
        return None

model = load_model()

# Header
st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 2rem;'>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])

# Sidebar inputs
with st.sidebar:
    st.markdown("### 📋 Enter Your Health Data")
    glucose = st.slider('🍯 Glucose (mg/dL)', 50, 300, 120)
    bp = st.slider('💓 Blood Pressure (mmHg)', 80, 200, 120)
    bmi = st.slider('⚖️ BMI', 15.0, 50.0, 25.0, 0.1)
    age = st.slider('🎂 Age (years)', 18, 100, 45)
    pregnancies = st.slider('🤰 Pregnancies', 0, 20, 2)
    skin_thickness = st.slider('🩸 Skin Thickness (mm)', 0, 100, 2)
    insulin = st.slider('💉 Insulin Level (µU/mL)', 0, 300, 100)
    diabetes_pedigree = st.slider('📊 Diabetes Pedigree Function', 0.0, 2.5, 0.5, 0.01)
    
    predict = st.button('🔮 Predict Risk', use_container_width=True)

# Display metrics
with col1:
    st.markdown("### 📊 Your Health Profile")
    metrics = [
        ("🍯 Glucose", f"{glucose} mg/dL", "Normal: 70-100"),
        ("💓 Blood Pressure", f"{bp} mmHg", "Normal: <120"),
        ("⚖️ BMI", f"{bmi:.1f}", "Normal: 18.5-24.9"),
        ("🎂 Age", f"{age} years", "Risk increases with age")
    ]
    for icon_name, value, normal in metrics:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{icon_name}</h3>
            <h2>{value}</h2>
            <small>{normal}</small>
        </div>
        """, unsafe_allow_html=True)

# Prediction results
with col2:
    st.markdown("### 🎯 Prediction Results")
    
    if predict and model:
        try:
            prediction = model.predict(np.array([[glucose, bp, bmi, age, pregnancies, insulin, skin_thickness, diabetes_pedigree]]))[0]
            
            if prediction == 1:
                st.markdown('<div class="high-risk">⚠️ HIGH RISK<br>Consult a doctor immediately!</div>', unsafe_allow_html=True)
                
                # Risk factors
                st.markdown("### 🔍 Risk Factors Found:")
                factors = []
                if glucose > 126: factors.append("🔴 High glucose level")
                if bp > 140: factors.append("🔴 High blood pressure") 
                if bmi >= 30: factors.append("🔴 Obesity")
                if age >= 65: factors.append("🔴 Advanced age")
                
                for factor in factors or ["⚠️ Multiple risk factors detected"]:
                    st.write(factor)
                    
            else:
                st.markdown('<div class="low-risk">✅ LOW RISK<br>Keep up the healthy lifestyle!</div>', unsafe_allow_html=True)
                
                # Healthy indicators
                st.markdown("### 🌟 Healthy Indicators:")
                healthy = []
                if glucose <= 100: healthy.append("✅ Normal glucose")
                if bp <= 120: healthy.append("✅ Normal blood pressure")
                if bmi < 25: healthy.append("✅ Normal weight")
                if age < 45: healthy.append("✅ Young age")
                
                for indicator in healthy or ["✅ Overall good health profile"]:
                    st.write(indicator)
            
            # Quick recommendations
            st.markdown("### 💡 Quick Tips:")
            tips = ["🥗 Eat balanced meals", "🏃‍♂️ Exercise regularly", "💧 Stay hydrated", "😴 Get enough sleep"]
            for tip in tips:
                st.write(tip)
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    elif predict and not model:
        st.error("❌ Model not found! Please check the model file.")
    
    else:
        st.info("👆 Click 'Predict Risk' to get your diabetes risk assessment!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.7;'>🏥 AI-Powered Health Assessment • Built with Streamlit</p>", unsafe_allow_html=True)
    
