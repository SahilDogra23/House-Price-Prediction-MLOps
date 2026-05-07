import streamlit as st
import requests

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California House Price Predictor")
st.markdown("Enter house details below to get an estimated price.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=52, value=25)
    median_income = st.number_input("Median Income (tens of thousands)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    bedrooms_per_room = st.number_input("Bedrooms per Room", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    ocean_proximity = st.selectbox("Ocean Proximity", 
                                   ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"])

with col2:
    latitude = st.number_input("Latitude", value=37.88)
    rooms_per_household = st.number_input("Rooms per Household", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    population_per_household = st.number_input("Population per Household", min_value=0.5, max_value=20.0, value=3.0, step=0.1)

st.divider()

# Convert ocean proximity to one-hot
ocean_map = {
    "<1H OCEAN":  [0, 0, 0, 0],
    "INLAND":     [1, 0, 0, 0],
    "ISLAND":     [0, 1, 0, 0],
    "NEAR BAY":   [0, 0, 1, 0],
    "NEAR OCEAN": [0, 0, 0, 1]
}

if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    inland, island, near_bay, near_ocean = ocean_map[ocean_proximity]
    
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "ocean_proximity_INLAND": inland,
        "ocean_proximity_ISLAND": island,
        "ocean_proximity_NEAR_BAY": near_bay,
        "ocean_proximity_NEAR_OCEAN": near_ocean
    }

    try:
        response = requests.post("http://127.0.0.1:8001/predict", json=payload)
        result = response.json()
        st.success(f"### Estimated House Price: {result['predicted_price']}")
        st.caption("This is an estimate based on 1990 California census data.")
    except Exception as e:
        st.error(f"Could not connect to API. Make sure uvicorn is running on port 8001!\n\n{e}")