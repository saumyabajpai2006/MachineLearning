import streamlit as st
import numpy as np
import joblib

#load the model
with open('models/NaiveBayes.pkl', 'rb') as file:
    model = joblib.load(file)
# Set the title of the app
st.set_page_config(page_title="Spam Detection App", page_icon="📧")
st.title("Spam Detection App")
st.subheader("Detect if an email is spam or not")

#sidebar for user input
st.sidebar.header("User Input")
def user_input_features():
    email_content = st.sidebar.text_area("Email Content", "Type your email content here...")
    return email_content

email_content = user_input_features()
#Load the vectorizer
with open('models/vectorizer.pkl', 'rb') as file:
    vectorizer = joblib.load(file)
# Predict button
if st.sidebar.button("Predict"):
    if email_content:
        # Preprocess the input
        input_data = np.array([email_content])
        # Make prediction
        prediction = model.predict(input_data)
        # Display the result
        if prediction[0] == 1:
            st.success("This email is classified as Spam.")
        else:
            st.success("This email is classified as Not Spam.")
    else:
        st.error("Please enter the email content to get a prediction.")
# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]")

