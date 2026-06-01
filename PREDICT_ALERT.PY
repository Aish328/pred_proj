import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD TRAINED PIPELINE
# =========================================================

with open("Patchtst.pkl", "rb") as f:
    nf = pickle.load(f)

print("\nPATCHTST MODEL LOADED SUCCESSFULLY!")

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

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

# =========================================================
# DATETIME
# =========================================================

df['Time'] = pd.to_datetime(
    df['Time'],
    format='%d/%m/%Y %H:%M:%S',
    errors='coerce'
)

df = df.dropna(subset=['Time'])

# =========================================================
# CLEAN TARGET
# =========================================================

df['Active Load'] = (
    df['Active Load']
    .astype(str)
    .str.extract(r'([-+]?\d*\.?\d+)')[0]
    .astype(float)
)

df = df.dropna(subset=['Active Load'])

# =========================================================
# CREATE PATCHTST FORMAT
# =========================================================

patch_df = pd.DataFrame()

patch_df['unique_id'] = (
    df['SUBSTATION'].astype(str)
    + "_"
    + df['FEEDER'].astype(str)
)

patch_df['ds'] = df['Time']

patch_df['y'] = df['Active Load']

# =========================================================
# SORT
# =========================================================

patch_df = patch_df.sort_values(
    ['unique_id', 'ds']
)

# =========================================================
# FORECAST
# =========================================================

print("\nGENERATING FORECASTS...\n")

forecast = nf.predict(df=patch_df)

print("\nFORECAST GENERATED!\n")

print(forecast.head())

# =========================================================
# RENAME FORECAST COLUMN
# =========================================================

forecast_col = forecast.columns[-1]

forecast = forecast.rename(
    columns={
        forecast_col: 'forecast'
    }
)

# =========================================================
# RISK SCORE
# =========================================================

forecast['risk_score'] = (
    forecast.groupby('unique_id')['forecast']
    .diff()
    .abs()
)

forecast['risk_score'] = (
    forecast['risk_score']
    .fillna(0)
)

# =========================================================
# ALERT LOGIC
# =========================================================

def classify_alert(row):

    pred = row['forecast']

    risk = row['risk_score']

    # impossible electrical prediction
    if pred < 0:
        return "CRITICAL"

    # sudden large jump
    elif risk > 10:
        return "HIGH RISK"

    # moderate instability
    elif risk > 5:
        return "WARNING"

    else:
        return "NORMAL"

forecast['alert'] = forecast.apply(
    classify_alert,
    axis=1
)

# =========================================================
# SHOW ALERT COUNTS
# =========================================================

print("\nALERT SUMMARY:\n")

print(
    forecast['alert']
    .value_counts()
)

# =========================================================
# SAVE ALERT REPORT
# =========================================================

forecast.to_csv(
    "alert_predictions.csv",
    index=False
)

print("\nALERT REPORT SAVED!")

# =========================================================
# SHOW ONLY CRITICAL ALERTS
# =========================================================

critical = forecast[
    forecast['alert'] == 'CRITICAL'
]

print("\nCRITICAL ALERTS:\n")

print(
    critical[
        ['unique_id',
         'ds',
         'forecast',
         'risk_score',
         'alert']
    ]
)

# =========================================================
# PLOT ONE FEEDER
# =========================================================

sample_feeder = forecast['unique_id'].iloc[0]

plot_df = forecast[
    forecast['unique_id'] == sample_feeder
]

plt.figure(figsize=(16,6))

plt.plot(
    plot_df['ds'],
    plot_df['forecast'],
    label='Forecast',
    linewidth=2
)

# highlight anomalies
critical_points = plot_df[
    plot_df['alert'] != 'NORMAL'
]

plt.scatter(
    critical_points['ds'],
    critical_points['forecast'],
    s=80
)

plt.title(
    f"Forecast + Predicted Alerts\n{sample_feeder}"
)

plt.xlabel("Time")

plt.ylabel("Forecasted Load")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "forecast_alerts.png",
    dpi=150
)

plt.show()

print("\nPLOT SAVED!")