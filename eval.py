# ANOMALY DETECTION USING LSTM AUTOENCODER

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataload import SmartGridDataLoader
from model import LSTMModel

# Load data again
filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\MAIIN_DATA (1).csv"

loader = SmartGridDataLoader(filepath)
loader.load_data()
loader.preprocess()
# loader.resample_hourly()

scaled = loader.scale_data()

X = loader.create_sequences(scaled, seq_length=48)
X = torch.tensor(X, dtype=torch.float32)
print("X_shape:", X.shape)
# split = int(0.8 * len(X))
# X_test = X[split:]
# # y_test = y[split:]

# X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Load model
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth",weights_only=True)) #loads trained weigts

model.eval() #switches to inference mode `  ```#disables : dropout, fixes batchnorm behaviour (for testing dropout and batch normalisation is not required)


with torch.no_grad():
    reconstructed = model(X)


errors = torch.mean((X-reconstructed)**2, dim=(1,2)).numpy()
print("\n Total Sequences: " , len(errors))

threshold = np.percentile(errors, 95)

anomalies= errors > threshold
print("Number of anomalies detected:", anomalies.sum()) 

print ( "threshold : " , threshold)

plt.figure(figsize=(14,5))

plt.plot(errors , label= "Reconstruction Errors", linewidth = 1.5)
plt.axhline ( threshold , color  = 'r', linestyle = '--', label = "Threshold")
plt.scatter(np.where(anomalies)[0],errors[anomalies], color = 'red', label = 'Anomalies', s = 10)

plt.title("LSTM Autoencoder ANomaly DEtection")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")

plt.legend()
plt.grid(alpha= 0.3)

plt.tight_layout()

plt.savefig("anomaly_detection_plot.png", dpi=300)
plt.show()
print("Anomaly detection plot saved as 'anomaly_detection_plot.png'")