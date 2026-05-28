import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    r"C:\Users\sharika\Desktop\pred\data\MAIIN_DATA (1).csv"
)
print(df.columns)
df['ds'] = pd.to_datetime(df['ds'])

df = df.sort_values(
    ['unique_id', 'ds']
)

print(df.head())

# =====================================================
# LOAD FULL PIPELINE
# =====================================================

with open(
    r"C:\Users\sharika\Desktop\pred\patchtst.pkl",
    "rb"
) as f:

    nf = pickle.load(f)

print("\nPIPELINE LOADED SUCCESSFULLY!")

# =====================================================
# FORECAST
# =====================================================

forecast = nf.predict()

print("\nFORECAST:\n")

print(forecast.head())

# =====================================================
# TEST SPLIT
# =====================================================

split = int(len(df) * 0.8)

test_df = df.iloc[split:]

# =====================================================
# RESET INDEX
# =====================================================

forecast = forecast.reset_index()

pred_col = forecast.columns[-1]

# =====================================================
# MERGE
# =====================================================

merged = pd.merge(
    test_df[['unique_id', 'ds', 'y']],
    forecast,
    on=['unique_id', 'ds'],
    how='inner'
)

print("\nMERGED SHAPE:", merged.shape)

# =====================================================
# METRICS
# =====================================================

y_true = merged['y']

y_pred = merged[pred_col]

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mape = np.mean(
    np.abs((y_true - y_pred) / y_true)
) * 100

print("\n======================")

print("PATCHTST RESULTS")

print("======================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"MAPE : {mape:.2f}%")

# =====================================================
# PLOT
# =====================================================

N = 200

plt.figure(figsize=(16,6))

plt.plot(
    y_true.values[:N],
    label="Actual",
    linewidth=2
)

plt.plot(
    y_pred.values[:N],
    label="Forecast",
    linewidth=2
)

plt.title("PatchTST Forecast vs Actual")

plt.xlabel("Time Steps")

plt.ylabel("Active Load")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()