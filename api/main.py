
import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the model packaged with the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "deployment", "model")
)
model = mlflow.pyfunc.load_model(MODEL_PATH)

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

    prediction = model.predict(input_data)

    return {
        "predicted_fare": round(float(prediction[0]), 2),
        "currency": "USD"
    }
