import torch
import numpy as np
import matplotlib.pyplot as plt
from dataload import SmartGridDataLoader
from anomaly import LSTMAutoEncoder
import pandas as pd
filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\MAIIN_DATA (1).csv"

loader = SmartGridDataLoader(filepath)
loader.load_data()
loader.preprocess()
loader.resample_hourly()

# loader.data_hourly['hour'] = loader.data_hourly.index.hour
# loader.data_hourly['day'] = loader.data_hourly.index.dayofweek

scaled_data = loader.scale_data()

X,y = loader.create_sequences(scaled_data)

X = torch.tensor(X,dtype=torch.float32)

model = LSTMAutoEncoder(input_size=X.shape[2])
model.load_state_dict(torch.load("anomaly_model.pth"))
model.eval()

with torch.no_grad():
    reconstructed = model(X)

errors = torch.mean((X-reconstructed)**2, dim = (1,2)).numpy()

errors_smooth = pd.Series(errors).rolling(window=50).mean()
errors_smooth = errors_smooth.bfill()
threshold = np.mean(errors)+3*np.std(errors)
anomalies = errors_smooth>threshold
anomaly_indices = np.where(anomalies)[0]
print("NUmber of anomalies", anomalies.sum())
plt.figure(figsize=(12,5))
plt.plot(errors_smooth, label="Reconstruction Error")
plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
# highlight anomalies
plt.scatter(anomaly_indices, errors_smooth.iloc[anomaly_indices], color='red', s=10, label="Anomalies")
print("Number of anomalies:", len(anomaly_indices))
plt.legend()
plt.title("Anomaly Detection")
plt.show()