import streamlit as st
import numpy as np
import joblib

# Load the trained model
with open('models/Linear_regression.pkl', 'rb') as file:
    model = joblib.load(file)

#title
st.set_page_config(page_title="House Price Predictor")
st.title('Price Prediction App')
st.subheader('Predict the price based on its features')

#sidebar
st.sidebar.header('Enter your details')
bedrooms = st.sidebar.slider('Bedrooms', 0, 10, 3)
bathrooms = st.sidebar.slider('Bathrooms', 0.0, 10.0, 2.0, step=0.25)
sqft_living = st.sidebar.slider('Living Area (sqft)', 500, 10000, 2500)
floors = st.sidebar.slider('Floors', 1, 4, 1)
waterfront = st.sidebar.selectbox('Waterfront View?', [0, 1], index=0)
view = st.sidebar.selectbox('View Quality (0-4)', [0, 1, 2, 3, 4], index=0)
grade = st.sidebar.selectbox('Grade (1-13)', list(range(1, 14)), index=6)
yr_built = st.sidebar.slider('Year Built', 1900, 2025, 1990)
#button to predict
if st.sidebar.button('Predict'):
    
    #Create input array
    features = np.array([[bedrooms, bathrooms, sqft_living, floors, waterfront, view, grade, yr_built]])
    predicted_price = model.predict(features)[0]

    # Display the result
    st.success(f'Predicted House Price: Rs. {predicted_price:,.2f}')
    st.info('This prediction is based on a trained regression model')
    
#footer
st.markdown("_ _ _ _ _ _ _ _ _ _ _ _ _ _")
st.markdown("Made with streamlit")
