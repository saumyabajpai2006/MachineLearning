import streamlit as st
import numpy as np
import joblib

# Load the trained model
with open('models/Linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)

#title
st.set_page_config(page_title="Price Predictor")
st.title('Price Predictor App')
st.subheader('Predict the price based on sqft_living, bedrooms, bathrooms, grade, zipcode')

#sidebar
st.sidebar.header('Enter your details')
experience = st.sidebar.slider('sqft_living', min_value=0, max_value=10000, step=100)
bedrooms = st.sidebar.slider('Bedrooms', min_value=0, max_value=10, step=1)
bathrooms = st.sidebar.slider('Bathrooms', min_value=0, max_value=10, step=0.5)
grade = st.sidebar.slider('Grade', min_value=1, max_value=13, step=1)
zipcode = st.sidebar.text_input('Zipcode', value='98101')


#button to predict
if st.sidebar.button('Predict'):
    # Convert zipcode to integer
    try:
        zipcode_int = int(zipcode)
        features = np.array([[experience, bedrooms, bathrooms, grade, zipcode_int]])
        price = model.predict(features)[0]
        st.success(f'Predicted Price: Rs. {price:,.2f}')
        st.info('This prediction is based on a simple linear regression model')
    except ValueError:
        st.error("Please enter a valid numeric zipcode.")
    
    
#footer
st.markdown("_ _ _ _ _ _ _ _ _ _ _ _ _ _")
st.markdown("Made with streamlit")
