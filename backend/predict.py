import torch
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from model import LSTMModel
from classifier import LSTMClassifier

from dataload import SmartGridDataLoader
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# =========================================
# DATABASE
# =========================================

DB_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/scada_db"

engine = create_engine(DB_URL)


# =========================================
# LOAD MODELS
# =========================================

autoencoder = LSTMModel(input_size=9)

autoencoder.load_state_dict(
    torch.load("lstm_model.pth", weights_only=True)
)

autoencoder.eval()


classifier = LSTMClassifier()

classifier.load_state_dict(
    torch.load("lstm_classifier.pth", weights_only=True)
)

classifier.eval()


FAULTS = {
    0: "No Fault",
    1: "Voltage Sag",
    2: "Current Imbalance",
    3: "Overload",
    4: "Voltage Swell",
    5: "Transformer Stress"
}


# =========================================
# MAIN PREDICTION FUNCTION
# =========================================

def run_prediction():

    # =====================================
    # FETCH DATA FROM POSTGRES
    # =====================================

    query = """
    SELECT *
    FROM feeder_data
    ORDER BY time DESC
    LIMIT 100
    """

    df = pd.read_sql(query, engine)

    df = df.sort_values("time")

    # =====================================
    # PREPROCESS
    # =====================================

    features = [
        'ir',
        'iy',
        'ib',
        'vry',
        'vyb',
        'vbr',
        'active_load'
    ]

    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['day'] = pd.to_datetime(df['time']).dt.dayofweek

    features.extend(['hour', 'day'])

    # scale manually
    data = df[features].values

    data = (data - data.min(axis=0)) / (
        data.max(axis=0) - data.min(axis=0) + 1e-8
    )

    # =====================================
    # CREATE SEQUENCE
    # =====================================

    seq = data[-48:]

    X = torch.tensor(
        seq,
        dtype=torch.float32
    ).unsqueeze(0)

    # =====================================
    # AUTOENCODER
    # =====================================

    with torch.no_grad():

        reconstructed = autoencoder(X)

    error = torch.mean(
        (X - reconstructed) ** 2
    ).item()

    threshold = 0.03

    anomaly = error > threshold

    # =====================================
    # CLASSIFIER
    # =====================================

    fault_name = "Normal"

    if anomaly:

        with torch.no_grad():

            output = classifier(X)

        pred = torch.argmax(output, dim=1).item()

        fault_name = FAULTS[pred]

    return {

        "anomaly": bool(anomaly),

        "reconstruction_error": float(error),

        "fault": fault_name
    }