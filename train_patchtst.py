import pickle
import logging
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

# =========================================================
# LOAD DATA
# =========================================================

filepath = r"C:\Users\sharika\Desktop\pred\data\MAIIN_DATA (1).csv"

print("Loading:", filepath)

df = pd.read_csv(
    filepath,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

print("\nOriginal Shape:", df.shape)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns)

# =========================================================
# PARSE TIMESTAMP
# Your format:
# 5/11/2026 1:06
# =========================================================

print("\nRaw Time values:")
print(df["Time"].head())

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%m/%d/%Y %H:%M",
    errors="coerce"
)

print("\nInvalid timestamps:", df["Time"].isna().sum())

df = df.dropna(subset=["Time"])

print("Shape after Time cleaning:", df.shape)

# =========================================================
# CLEAN ACTIVE LOAD
# Example:
# 1.53 MW
# =========================================================

print("\nRaw Active Load values:")
print(df["Active Load"].head())

df["Active Load"] = (
    df["Active Load"]
    .astype(str)
    .str.replace("\xa0", "", regex=False)  # remove hidden spaces
    .str.replace("MW", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

print("\nCleaned Active Load strings:")
print(df["Active Load"].head())

df["Active Load"] = pd.to_numeric(
    df["Active Load"],
    errors="coerce"
)

print("\nInvalid Active Load values:",
      df["Active Load"].isna().sum())

df = df.dropna(subset=["Active Load"])

print("Shape after Active Load cleaning:", df.shape)

# =========================================================
# OPTIONAL OUTLIER CLIPPING
# =========================================================

q1 = df["Active Load"].quantile(0.01)
q99 = df["Active Load"].quantile(0.99)

df["Active Load"] = df["Active Load"].clip(q1, q99)

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

print("\nNull counts:")
print(
    df[
        ["SUBSTATION",
         "FEEDER",
         "Time",
         "Active Load"]
    ].isna().sum()
)

# =========================================================
# CREATE PATCHTST DATASET
# =========================================================

patch_df = pd.DataFrame()

patch_df["unique_id"] = (
    df["SUBSTATION"].astype(str).str.strip()
    + "_"
    + df["FEEDER"].astype(str).str.strip()
)

patch_df["ds"] = df["Time"]

patch_df["y"] = df["Active Load"]

print("\nBefore dropna:", patch_df.shape)

patch_df = patch_df.dropna()

print("After dropna:", patch_df.shape)

patch_df = patch_df.sort_values(
    by=["unique_id", "ds"]
).reset_index(drop=True)

print("\nFINAL DATAFRAME:")
print(patch_df.head())

print("\nShape:", patch_df.shape)

print("\nNumber of Series:",
      patch_df["unique_id"].nunique())

print("\nSeries Lengths:")
print(
    patch_df.groupby("unique_id")
    .size()
    .sort_values()
)

# =========================================================
# SAVE CLEANED DATA
# =========================================================

patch_df.to_csv(
    "patchtst_data.csv",
    index=False
)

print("\npatchtst_data.csv saved!")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

# ==================================================
# Train on complete dataset
# ==================================================

train_df = patch_df.copy()

print("\nTraining Shape:", train_df.shape)

print(
    train_df.groupby("unique_id")
    .size()
)

if len(train_df) == 0:
    raise ValueError(
        "Training dataframe is empty. "
        "Check preprocessing."
    )

# =========================================================
# MODEL
# =========================================================

logging.getLogger(
    "pytorch_lightning"
).setLevel(logging.INFO)

model = PatchTST(
    h=6,
    input_size=96,
    patch_len=24,
    stride=12,
    hidden_size=128,
    n_heads=8,
    batch_size=16,
    learning_rate=0.0005,
    max_steps=150,
    val_check_steps=10,

    # Use identity since you've already cleaned the data
    scaler_type="identity"
)

nf = NeuralForecast(
    models=[model],
    freq="3min"
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


with open("Patchtst.pkl", "wb") as f:
    pickle.dump(nf, f)

print("\nPatchTST model saved!")



forecast = nf.predict()

print("\nForecast:")
print(forecast.head())

forecast.to_csv(
    "patchtst_forecast.csv",
    index=False
)

print("\nForecast saved!")