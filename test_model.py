import torch
import numpy as np
import matplotlib.pyplot as plt
from dataload import SmartGridDataLoader
from model import LSTMModel

filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\synthetic_smartgrid_test_data.csv"
loader = SmartGridDataLoader(filepath)
loader.load_data()
loader.preprocess()

scaled = loader.scale_data()
X = loader.create_sequences(scaled, seq_length=48)

X = torch.tensor(X, dtype=torch.float32)

print("input shape:", X.shape)

model = LSTMModel(input_size=X.shape[2])

model.load_state_dict(torch.load("lstm_model.pth", weights_only=True)) #loads trained weights
model.eval()

print("Model loaded successfully. Starting evaluation...")
with torch.no_grad():

    reconstructed = model(X)

errors = torch.mean(
    (X - reconstructed) ** 2,
    dim=(1, 2)
).detach().numpy()

threshold = np.percentile(errors, 95)

anomalies = errors > threshold

print("\nTotal anomalies detected:", anomalies.sum())


# =========================================================
# ANOMALY INDICES
# =========================================================

anomaly_indices = np.where(anomalies)[0]

print("\nDetected anomaly positions:\n")
print(anomaly_indices)


# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(16, 6))

plt.plot(
    errors,
    label="Reconstruction Error",
    linewidth=1.5
)

plt.axhline(
    threshold,
    color='red',
    linestyle='--',
    label='Threshold'
)

plt.scatter(
    anomaly_indices,
    errors[anomaly_indices],
    color='red',
    label='Detected Anomalies'
)

plt.title("Smart Grid Anomaly Detection")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "test_anomaly_results.png",
    dpi=150
)

plt.show()

print("\nPlot saved as test_anomaly_results.png")