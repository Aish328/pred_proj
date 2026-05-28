import pickle
import logging
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

# =========================================================
# LOAD DATA
# =========================================================

filepath = r"C:\Users\sharika\Desktop\pred\data\MAIIN_DATA (1).csv"

df = pd.read_csv(
    filepath,
    encoding='latin1',
    engine='python',
    on_bad_lines='skip'
)

print("\nOriginal Shape:", df.shape)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

print("\nColumns:\n")

print(df.columns)

# =========================================================
# CONVERT TIME COLUMN
# =========================================================

df['Time'] = pd.to_datetime(
    df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)

df = df.dropna(subset=['Time'])

# =========================================================
# CLEAN ACTIVE LOAD
# =========================================================

df['Active Load'] = (
    df['Active Load']
    .astype(str)
    .str.extract(r'([-+]?\d*\.?\d+)')[0]
    .astype(float)
)

df = df.dropna(subset=['Active Load'])

# =========================================================
# REMOVE EXTREME OUTLIERS
# =========================================================

q1 = df['Active Load'].quantile(0.01)

q99 = df['Active Load'].quantile(0.99)

df['Active Load'] = df['Active Load'].clip(q1, q99)

# =========================================================
# CREATE PATCHTST DATAFRAME
# =========================================================

patch_df = pd.DataFrame()

# ---------------------------------------------------------
# MULTI-SERIES IDENTIFIER
# ---------------------------------------------------------

patch_df['unique_id'] = (
    df['SUBSTATION'].astype(str)
    + "_"
    + df['FEEDER'].astype(str)
)

# ---------------------------------------------------------
# TIMESTAMP
# ---------------------------------------------------------

patch_df['ds'] = df['Time']

# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

patch_df['y'] = df['Active Load']

# =========================================================
# REMOVE NaNs
# =========================================================

patch_df = patch_df.dropna()

# =========================================================
# SORT DATA
# =========================================================

patch_df = patch_df.sort_values(
    by=['unique_id', 'ds']
)

# =========================================================
# FINAL CHECK
# =========================================================

print("\nFINAL DATAFRAME:\n")

print(patch_df.head())

print("\nShape:", patch_df.shape)

print("\nUnique IDs:\n")

print(patch_df['unique_id'].unique())

# =========================================================
# SAVE CLEAN DATASET
# =========================================================

patch_df.to_csv(
    "patchtst_data.csv",
    index=False
)

print("\npatchtst_data.csv saved!")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(len(patch_df) * 0.8)

train_df = patch_df.iloc[:split]

test_df = patch_df.iloc[split:]

print("\nTrain Shape:", train_df.shape)

print("Test Shape:", test_df.shape)

# =========================================================
# ENABLE EPOCH LOGGING
# =========================================================

logging.getLogger(
    "pytorch_lightning"
).setLevel(logging.INFO)

# =========================================================
# PATCHTST MODEL
# =========================================================

model = PatchTST(

    # -----------------------------------------------------
    # Forecast Horizon
    # -----------------------------------------------------

    h=6,

    # -----------------------------------------------------
    # Lookback Window
    # -----------------------------------------------------

    input_size=288,

    # -----------------------------------------------------
    # Patch Parameters
    # -----------------------------------------------------

    patch_len=24,

    stride=12,

    # -----------------------------------------------------
    # Transformer Size
    # -----------------------------------------------------

    hidden_size=128,

    n_heads=8,

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    batch_size=16,

    learning_rate=0.0005,

    max_steps=150,

    val_check_steps=10,

    scaler_type='standard'
)

# =========================================================
# NEURAL FORECAST
# =========================================================

nf = NeuralForecast(
    models=[model],
    freq='3min'
)

# =========================================================
# TRAIN
# =========================================================

print("\n==============================")

print("PATCHTST TRAINING STARTED")

print("==============================\n")

nf.fit(df=train_df)

print("\n==============================")

print("TRAINING COMPLETE")

print("==============================")

# =========================================================
# SAVE FULL PIPELINE
# =========================================================

with open("Patchtst.pkl", "wb") as f:

    pickle.dump(nf, f)

print("\nFull NeuralForecast Pipeline Saved!")

# =========================================================
# SAMPLE FORECAST
# =========================================================

forecast = nf.predict()

print("\nForecast Head:\n")

print(forecast.head())

# =========================================================
# SAVE FORECAST
# =========================================================

forecast.to_csv(
    "patchtst_forecast.csv",
    index=False
)

print("\nForecast saved as patchtst_forecast.csv")