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

# Define input fields for features based on your columns
id = st.number_input("ID", value=1)  # Adjust default value as needed
age = st.number_input("Age")
flight_distance = st.number_input("Flight Distance")
inflight_wifi_service = st.number_input("Inflight Wifi Service", min_value=0, max_value=5)  # Assuming a scale of 0 to 5
departure_arrival_time_convenient = st.number_input("Departure/Arrival Time Convenient", min_value=0, max_value=5)
ease_of_online_booking = st.number_input("Ease of Online Booking", min_value=0, max_value=5)
gate_location = st.number_input("Gate Location", min_value=0, max_value=5)
food_and_drink = st.number_input("Food and Drink", min_value=0, max_value=5)
online_boarding = st.number_input("Online Boarding", min_value=0, max_value=5)
seat_comfort = st.number_input("Seat Comfort", min_value=0, max_value=5)
inflight_entertainment = st.number_input("Inflight Entertainment", min_value=0, max_value=5)
onboard_service = st.number_input("On-board Service", min_value=0, max_value=5)
leg_room_service = st.number_input("Leg Room Service", min_value=0, max_value=5)
baggage_handling = st.number_input("Baggage Handling", min_value=0, max_value=5)
checkin_service = st.number_input("Checkin Service", min_value=0, max_value=5)
inflight_service = st.number_input("Inflight Service", min_value=0, max_value=5)
cleanliness = st.number_input("Cleanliness", min_value=0, max_value=5)

# Button to make prediction
if st.button("Get Prediction"):
    # Assemble the input features into a DataFrame
    input_data = pd.DataFrame({
        'ID': [id],
        'Age': [age],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [inflight_wifi_service],
        'Departure/Arrival time convenient': [departure_arrival_time_convenient],
        'Ease of Online booking': [ease_of_online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_and_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
    })

    # Make prediction
    prediction = rf.predict(input_data)

    # Display the result
    st.write("Prediction: ", prediction[0])

# Run the application
if __name__ == '__main__':
    st.write("Please enter the features above to get predictions.")
