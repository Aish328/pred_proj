"""
train_patchtst.py
---------------------------------------------------------
PatchTST Training Script for SCADA Active Load Forecasting
Fixes:
  - h=480 (24 hours ahead at 3min freq)
  - input_size=960 (48 hours context)
  - max_steps=1000
  - No scaling (matches dashboard's unscaled historical plot)
  - Anchored forecast start to last historical timestamp
  - Saves anchor timestamp alongside model for inference sync
---------------------------------------------------------
"""

import pickle
import logging
import pandas as pd
import numpy as np
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

print("\nColumns:", df.columns.tolist())

# =========================================================
# PARSE TIMESTAMP
# =========================================================

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%m/%d/%Y %H:%M",
    errors="coerce"
)

print("Invalid timestamps:", df["Time"].isna().sum())
df = df.dropna(subset=["Time"])
print("Shape after Time cleaning:", df.shape)

# =========================================================
# CLEAN ACTIVE LOAD
# =========================================================

df["Active Load"] = (
    df["Active Load"]
    .astype(str)
    .str.replace("\xa0", "", regex=False)
    .str.replace("MW", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Active Load"] = pd.to_numeric(df["Active Load"], errors="coerce")

print("Invalid Active Load values:", df["Active Load"].isna().sum())
df = df.dropna(subset=["Active Load"])
print("Shape after Active Load cleaning:", df.shape)

# =========================================================
# OUTLIER CLIPPING
# =========================================================

q1  = df["Active Load"].quantile(0.01)
q99 = df["Active Load"].quantile(0.99)
df["Active Load"] = df["Active Load"].clip(q1, q99)

print(f"\nLoad range after clipping: {q1:.2f} – {q99:.2f} MW")

# =========================================================
# BUILD PATCHTST DATAFRAME
# =========================================================

patch_df = pd.DataFrame({
    "unique_id": (
        df["SUBSTATION"].astype(str).str.strip()
        + "_"
        + df["FEEDER"].astype(str).str.strip()
    ),
    "ds": df["Time"],
    "y":  df["Active Load"],
}).dropna()

patch_df = patch_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

print("\nFINAL DATAFRAME:")
print(patch_df.head())
print("\nShape:", patch_df.shape)
print("\nNumber of Series:", patch_df["unique_id"].nunique())
print("\nSeries Lengths:")
print(patch_df.groupby("unique_id").size().sort_values())

# =========================================================
# DATA QUALITY CHECK
# =========================================================

print("\n--- Data Quality ---")
print("Date range:", patch_df["ds"].min(), "→", patch_df["ds"].max())
total_days = (patch_df["ds"].max() - patch_df["ds"].min()).days
print(f"Total days: {total_days}")
print("Load stats:\n", patch_df["y"].describe())

if total_days < 7:
    print("\n⚠️  WARNING: Less than 7 days of data — model may not learn daily patterns well.")
    print("   Recommend at least 2 weeks of data for reliable 24hr forecasting.")

# Check for gaps
for uid, grp in patch_df.groupby("unique_id"):
    grp = grp.sort_values("ds")
    gaps = grp["ds"].diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta("10min")]
    if len(large_gaps) > 0:
        print(f"\n⚠️  Series {uid} has {len(large_gaps)} gaps > 10 min")
        print(large_gaps.value_counts().head())

# =========================================================
# SAVE CLEANED DATA
# =========================================================

patch_df.to_csv("patchtst_data.csv", index=False)
print("\npatchtst_data.csv saved!")

# =========================================================
# SAVE ANCHOR TIMESTAMPS (for inference sync)
# Last timestamp per series — forecast must start right after this
# =========================================================

anchor_timestamps = (
    patch_df.groupby("unique_id")["ds"]
    .max()
    .to_dict()
)
print("\nAnchor timestamps (last known per series):")
for k, v in anchor_timestamps.items():
    print(f"  {k}: {v}")

# =========================================================
# TRAIN
# =========================================================

if len(patch_df) == 0:
    raise ValueError("Training dataframe is empty. Check preprocessing.")

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

model = PatchTST(
    h=480,              # 24 hours ahead (480 × 3min = 1440min = 24hr)
    input_size=960,     # 48 hours context (960 × 3min = 48hr)
    patch_len=16,       # 48 min per patch — good for 3min data
    stride=8,
    hidden_size=256,    # increased capacity
    n_heads=8,
    batch_size=32,
    learning_rate=0.0001,
    max_steps=1000,
    val_check_steps=50,
    scaler_type="identity",   # no scaling — matches unscaled dashboard
    dropout=0.1,
)

nf = NeuralForecast(
    models=[model],
    freq="3min"
)

print("\n==============================")
print("PATCHTST TRAINING STARTED")
print(f"  Horizon:    480 steps = 24 hours")
print(f"  Context:    960 steps = 48 hours")
print(f"  Frequency:  3 minutes")
print(f"  Steps:      1000")
print("==============================\n")

nf.fit(df=patch_df)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

# =========================================================
# SAVE MODEL + ANCHOR TIMESTAMPS TOGETHER
# =========================================================

save_bundle = {
    "model":             nf,
    "anchor_timestamps": anchor_timestamps,
    "freq":              "3min",
    "h":                 480,
    "trained_on":        pd.Timestamp.now().isoformat(),
    "series_ids":        patch_df["unique_id"].unique().tolist(),
}

with open("Patchtst.pkl", "wb") as f:
    pickle.dump(save_bundle, f)

print("\nPatchTST bundle saved to Patchtst.pkl")
print("Bundle keys:", list(save_bundle.keys()))

# =========================================================
# VERIFY FORECAST — must start right after last history point
# =========================================================

forecast = nf.predict()
forecast = forecast.reset_index()

print("\nForecast sample:")
print(forecast.head(10))
print("\nForecast shape:", forecast.shape)

pred_col = [c for c in forecast.columns if c not in ("unique_id", "ds")][0]

# Verify continuity for each series
print("\n--- Continuity Check ---")
for uid, grp in forecast.groupby("unique_id"):
    anchor = anchor_timestamps.get(uid)
    first_forecast_dt = grp["ds"].min()
    gap = first_forecast_dt - anchor
    print(f"  {uid}:")
    print(f"    Last history:     {anchor}")
    print(f"    First forecast:   {first_forecast_dt}")
    print(f"    Gap:              {gap}  ← should be 3min")
    print(f"    Forecast range:   {grp['ds'].min()} → {grp['ds'].max()}")
    print(f"    Values (first 5): {grp[pred_col].head().values.round(2)}")

forecast.to_csv("patchtst_forecast.csv", index=False)
print("\nForecast saved to patchtst_forecast.csv!")