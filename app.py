from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = FastAPI(title="House Price Prediction API")

def train_model():
    df = pd.read_csv("data/housing.csv")
    df = df[df["median_house_value"] < 500001].copy()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df = df[df["rooms_per_household"] < 10]
    df = df[df["population_per_household"] < 20]
    df["price_log"] = np.log(df["median_house_value"])
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    features_to_drop = ["price_log", "median_house_value", "total_rooms",
                        "total_bedrooms", "population", "households"]
    X = df.drop(columns=features_to_drop)
    y = df["price_log"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_features="log2", max_depth = 10)
    model.fit(X_scaled, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_model.pkl")
    joblib.dump(scaler, "models/house_price_scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/house_price_features.pkl")
    print("Model trained and saved!")

if not os.path.exists("models/house_price_model.pkl"):
    train_model()

model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/house_price_scaler.pkl")
feature_names = joblib.load("models/house_price_features.pkl")

class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    median_income: float
    rooms_per_household: float
    bedrooms_per_room: float
    population_per_household: float
    ocean_proximity_INLAND: int
    ocean_proximity_ISLAND: int
    ocean_proximity_NEAR_BAY: int = 0
    ocean_proximity_NEAR_OCEAN: int = 0

@app.get("/")
def root():
    return {"status": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(data: HouseData):
    input_array = np.array([[
        data.longitude, data.latitude, data.housing_median_age,
        data.median_income, data.rooms_per_household, data.bedrooms_per_room,
        data.population_per_household, data.ocean_proximity_INLAND,
        data.ocean_proximity_ISLAND, data.ocean_proximity_NEAR_BAY,
        data.ocean_proximity_NEAR_OCEAN
    ]])
    input_scaled = scaler.transform(input_array)
    price_log = model.predict(input_scaled)[0]
    price = np.exp(price_log)
    return {
        "predicted_price": f"${price:,.0f}",
        "price_raw": round(float(price), 2)
    }