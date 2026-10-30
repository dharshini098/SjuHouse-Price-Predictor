import streamlit as st
import joblib

# Loading the trained model
model = joblib.load("house_model.pkl")

st.title("House Price Predictor App")
st.write("Enter House Details to Predict Price: ")

size = st.number_input("House size (Sq.ft) : ")
bedrooms = st.number_input("Number of Bedrooms : ")
age = st.number_input("Age of House : ")

# Prediction Button
if st.button("Predict Price : "):
    prediction = model.predict([[size,bedrooms,age]])
    st.success(f"Estimated Price : ${prediction[0]:,.2f}")
    