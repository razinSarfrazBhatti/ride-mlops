
import os
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_STORE = os.getenv("MODEL_STORE", "/models")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

MODEL_PATH = os.path.join(MODEL_STORE, MODEL_VERSION)

app = FastAPI(
    title="Ride Fare Prediction API",
    description="Predict taxi fares using a versioned ML model",
    version="1.1.0"
)


@lru_cache(maxsize=1)
def get_model():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model version {MODEL_VERSION} not found at {MODEL_PATH}"
        )

    return mlflow.pyfunc.load_model(MODEL_PATH)


class TripRequest(BaseModel):
    trip_distance: float = Field(gt=0, le=100)
    trip_duration: float = Field(gt=0, le=180)
    pickup_hour: int = Field(ge=0, le=23)
    passenger_count: float = Field(ge=1, le=8)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict_fare(request: TripRequest):
    input_data = pd.DataFrame([{
        "trip_distance": request.trip_distance,
        "trip_duration": request.trip_duration,
        "pickup_hour": request.pickup_hour,
        "passenger_count": request.passenger_count
    }])

    try:
        prediction = get_model().predict(input_data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed"
        ) from exc

    return {
        "predicted_fare": round(float(prediction[0]), 2),
        "currency": "USD",
        "model_version": MODEL_VERSION
    }
