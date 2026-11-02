
import os
from functools import lru_cache

import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Connect to our MLflow tracking server
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)


@lru_cache(maxsize=1)
def get_model():
    """Load the registered model once and reuse it."""
    return mlflow.pyfunc.load_model(
        "models:/ride-fare-predictor/1"
    )


app = FastAPI(
    title="Ride Fare Prediction API",
    description="Predict taxi fares using a trained ML model",
    version="1.0.0"
)


class TripRequest(BaseModel):
    trip_distance: float = Field(gt=0, le=100)
    trip_duration: float = Field(gt=0, le=180)
    pickup_hour: int = Field(ge=0, le=23)
    passenger_count: float = Field(ge=1, le=8)


@app.get("/")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_fare(request: TripRequest):
    input_data = pd.DataFrame([{
        "trip_distance": request.trip_distance,
        "trip_duration": request.trip_duration,
        "pickup_hour": request.pickup_hour,
        "passenger_count": request.passenger_count
    }])

    model = get_model()
    prediction = model.predict(input_data)

    return {
        "predicted_fare": round(float(prediction[0]), 2),
        "currency": "USD"
    }
