import streamlit as st
import numpy as np
import joblib

# Load the trained model
with open('models/simple_linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)

#title
st.set_page_config(page_title="Salary Predictor")
st.title('Salary Predictor App')
st.subheader('Predict the salary based on years of experience')

#sidebar
st.sidebar.header('Enter your details')
experience = st.sidebar.slider('Years of Experience', min_value=0.0, max_value=20.0, step=0.5)

#button to predict
if st.sidebar.button('Predict'):
    # Predict the salary
    salary = model.predict(np.array([[experience]]))[0]
    
    # Display the result
    st.success(f'Predicted Salary: Rs. {salary:,.2f}') #success is green colour

    #additional information
    st.info('This prediction is based on a simple linear regression model')

#footer
st.markdown("_ _ _ _ _ _ _ _ _ _ _ _ _ _")
st.markdown("Made with streamlit")
