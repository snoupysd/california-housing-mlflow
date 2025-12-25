import streamlit as st
import requests

st.title("California Housing - API Client")

api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

payload = {
    "MedInc": st.number_input("MedInc", value=5.0),
    "HouseAge": st.number_input("HouseAge", value=30.0),
    "AveRooms": st.number_input("AveRooms", value=6.0),
    "AveBedrms": st.number_input("AveBedrms", value=2),
    "Population": st.number_input("Population", value=800.0),
    "AveOccup": st.number_input("AveOccup", value=2),
    "Latitude": st.number_input("Latitude", value=14),
    "Longitude": st.number_input("Longitude", value=-50),
}

if st.button("Predict"):
    r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    if r.status_code == 200:
        st.success(f"Prediction: {r.json()['prediction']:.4f}")
    else:
        st.error(f"Error {r.status_code}")
        st.code(r.text)
