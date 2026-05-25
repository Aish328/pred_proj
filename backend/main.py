from fastapi import FastAPI
from backend.predict import run_prediction
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "Smart Grid AI Backend Running"
    }


@app.get("/predict")
def predict():

    result = run_prediction()

    return result