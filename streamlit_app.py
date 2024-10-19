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

# Define input fields for the selected features
inflight_wifi_service = st.number_input("Inflight Wifi Service", min_value=0, max_value=5)
seat_comfort = st.number_input("Seat Comfort", min_value=0, max_value=5)
food_and_drink = st.number_input("Food and Drink", min_value=0, max_value=5)
inflight_entertainment = st.number_input("Inflight Entertainment", min_value=0, max_value=5)
cleanliness = st.number_input("Cleanliness", min_value=0, max_value=5)

# Button to make prediction
if st.button("Get Prediction"):
    # Assemble the input features into a DataFrame
    input_data = pd.DataFrame({
        'Inflight wifi service': [inflight_wifi_service],
        'Seat comfort': [seat_comfort],
        'Food and drink': [food_and_drink],
        'Inflight entertainment': [inflight_entertainment],
        'Cleanliness': [cleanliness],
    })

    # Display the input data for debugging
    st.write("Input Data for Prediction:")
    st.write(input_data)

    # Make prediction
    try:
        prediction = rf.predict(input_data)
        # Display the result
        st.write("Prediction: ", prediction[0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Run the application
if __name__ == '__main__':
    st.write("Please enter the features above to get predictions.")

