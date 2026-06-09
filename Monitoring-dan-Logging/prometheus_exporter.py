from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

import joblib
import pandas as pd
import time

app = FastAPI()

# ====================================
# LOAD MODEL
# ====================================

model = joblib.load("titanic_model.pkl")

# ====================================
# PROMETHEUS METRICS
# ====================================

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

SUCCESS_COUNT = Counter(
    "successful_predictions_total",
    "Successful predictions"
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Prediction errors"
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

PREDICTION_VALUE = Gauge(
    "last_prediction",
    "Last prediction result"
)

# ====================================
# INPUT SCHEMA
# ====================================

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

# ====================================
# PREDICT ENDPOINT
# ====================================

@app.post("/predict")
def predict(data: Passenger):

    REQUEST_COUNT.inc()

    start_time = time.time()

    try:

        df = pd.DataFrame([{
            "Pclass": data.Pclass,
            "Sex": data.Sex,
            "Age": data.Age,
            "SibSp": data.SibSp,
            "Parch": data.Parch,
            "Fare": data.Fare,
            "Embarked": data.Embarked
        }])

        prediction = int(model.predict(df)[0])

        SUCCESS_COUNT.inc()

        PREDICTION_VALUE.set(prediction)

        LATENCY.observe(
            time.time() - start_time
        )

        return {
            "prediction": prediction
        }

    except Exception as e:

        ERROR_COUNT.inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ====================================
# METRICS ENDPOINT
# ====================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )