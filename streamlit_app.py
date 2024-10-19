# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model
model_path = "compressed_rf_model.pkl"  # Ensure this path matches your uploaded model file location

try:
    rf = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model is uploaded and the path is correct.")

# Set up the interface
st.title("Target Value Prediction using Random Forest Model")
st.write("Enter the features to get predictions.")

# Define input fields for features
# Ensure these names match the column names in your data
feature_1 = st.number_input("Feature 1")
feature_2 = st.number_input("Feature 2")
feature_3 = st.number_input("Feature 3")
feature_4 = st.number_input("Feature 4")

# Add more features as needed
# For example:
# feature_n = st.number_input("Feature N")

# Button to make prediction
if st.button("Get Prediction"):
    # Assemble the input features into a DataFrame
    input_data = pd.DataFrame({
        'feature_1': [feature_1],
        'feature_2': [feature_2],
        'feature_3': [feature_3],
        'feature_4': [feature_4],
        # 'feature_n': [feature_n]  # Add other features here
    })

    # Make prediction
    prediction = rf.predict(input_data)

    # Display the result
    st.write("Prediction: ", prediction[0])

# Run the application
if __name__ == '__main__':
    st.write("Please enter the features above to get predictions.")
