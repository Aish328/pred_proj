# Run this locally to see your actual data distribution
# Copy output and share it here

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

filepath = r"C:\Users\sharika\Desktop\Pred\data\MAIIN_DATA (1).csv"

df = pd.read_csv(filepath, sep=',', encoding='latin1', engine='python',
                 na_values=['?'], on_bad_lines='skip')
df.columns = df.columns.str.strip()
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df = df.dropna(subset=['Time']).set_index('Time')

features = ['IR', 'IY', 'IB', 'VRY', 'VYB', 'VBR', 'Active Load']
for col in features:
    df[col] = df[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0].astype(float)
df = df.ffill()
for col in features:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

print("=== RAW (unscaled) statistics ===")
print(df[features].describe().round(3).to_string())

scaler = MinMaxScaler()
df.columns = df.columns.str.strip()
feats_scale = ['IR', 'IY', 'IB', 'VRY', 'VYB', 'VBR', 'Active Load']
df['hour'] = df.index.hour
df['day']  = df.index.dayofweek
all_feats  = feats_scale + ['hour', 'day']
scaled = scaler.fit_transform(df[all_feats])
s = pd.DataFrame(scaled, columns=all_feats)

print("\n=== SCALED statistics ===")
print(s[feats_scale].describe().round(3).to_string())

print("\n=== SCALED percentiles ===")
for col in feats_scale:
    vals = s[col]
    print(f"{col:12s}  p1={vals.quantile(0.01):.3f}  p5={vals.quantile(0.05):.3f}"
          f"  p10={vals.quantile(0.10):.3f}  p90={vals.quantile(0.90):.3f}"
          f"  p95={vals.quantile(0.95):.3f}  p99={vals.quantile(0.99):.3f}")

print("\n=== 3-phase current imbalance distribution ===")
IR, IY, IB = s['IR'], s['IY'], s['IB']
mean_I  = (IR + IY + IB) / 3
imbal   = pd.concat([
    (IR - mean_I).abs(),
    (IY - mean_I).abs(),
    (IB - mean_I).abs()
], axis=1).max(axis=1)
print(imbal.describe().round(4).to_string())
print(f"  % rows with imbal > 0.05 : {(imbal>0.05).mean()*100:.1f}%")
print(f"  % rows with imbal > 0.10 : {(imbal>0.10).mean()*100:.1f}%")
print(f"  % rows with imbal > 0.15 : {(imbal>0.15).mean()*100:.1f}%")

print("\n=== Active Load distribution ===")
for t in [0.60, 0.65, 0.70, 0.75, 0.80]:
    print(f"  % rows with Load > {t} : {(s['Active Load']>t).mean()*100:.1f}%")