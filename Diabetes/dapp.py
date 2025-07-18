import streamlit as st
import numpy as np
import joblib

# Load the trained model
with open('models/xgboost.pkl', 'rb') as file:
    model = joblib.load(file)

# Title
st.set_page_config(page_title="Diabetes Predictor")
st.title('Diabetes Predictor App')
st.subheader('Predict the diabetes risk based on health metrics')

# Sidebar
st.sidebar.header('Enter your details')
glucose = st.sidebar.slider('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.sidebar.slider('Blood Pressure', min_value=0, max_value=150, step=1)
bmi = st.sidebar.slider('BMI', min_value=0.0, max_value=50.0, step=0.1)
age = st.sidebar.slider('Age', min_value=0, max_value=120, step=1)
pregnancies = st.sidebar.slider('Pregnancies', min_value=0, max_value=20, step=1)
skin_thickness = st.sidebar.slider('Skin Thickness', min_value=0, max_value=100, step=1)
insulin = st.sidebar.slider('Insulin Level', min_value=0, max_value=300, step=1)
diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)


# Button to predict
if st.sidebar.button('Predict'):
    # Prepare input data
    input_data = np.array([[glucose, blood_pressure, bmi, age, pregnancies, skin_thickness, insulin, diabetes_pedigree]])
    
    # Predict the diabetes risk
    prediction = model.predict(input_data)[0]
    
    # Display the result
    if prediction == 1:
        st.success('Predicted: High Risk of Diabetes')
    else:
        st.success('Predicted: Low Risk of Diabetes')

    # Additional information
    st.info('This prediction is based on an XGBoost model trained on health metrics.')

# Footer
st.markdown("_ _ _ _ _ _ _ _ _ _ _ _ _ _")
st.markdown("Made with Streamlit")