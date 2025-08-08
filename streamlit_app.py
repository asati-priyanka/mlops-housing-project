import streamlit as st
import requests

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 Housing Price Predictor")

# --- API target selector ---
target = st.selectbox(
    "API Target",
    ["Localhost (127.0.0.1:8000)", "Docker (api:8000)"],
    index=0,
)
API_URL = "http://127.0.0.1:8000/predict" if target.startswith("Localhost") else "http://api:8000/predict"
st.caption(f"Calling: **{API_URL}**")

# --- Input form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, value=8.3252, format="%.4f")
        HouseAge = st.number_input("House Age", min_value=0.0, value=41.0, format="%.1f")
        AveRooms = st.number_input("Average Rooms", min_value=0.0, value=6.9841, format="%.4f")
        AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0238, format="%.4f")
    with col2:
        Population = st.number_input("Population", min_value=0.0, value=322.0, format="%.1f")
        AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=2.5556, format="%.4f")
        Latitude = st.number_input("Latitude", value=37.88, format="%.2f")
        Longitude = st.number_input("Longitude", value=-122.23, format="%.2f")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "records": [{
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }]
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            pred = resp.json().get("predictions", [None])[0]
            st.success(f"✅ Predicted Median House Value: **{pred:.4f}**")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
