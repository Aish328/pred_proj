import torch
import numpy as np
from model import LSTMModel
from dataload import SmartGridDataLoader
from classifier_model import LSTMClassifier
import matplotlib.pyplot as plt
FAULTS  = {
    0: "No Fault",
    1: "VOLTAGE SAG",
    2: "CURRENT IMBALANCE",
    3: "OVERLOAD",
    4: "VOLTAGE SWELL",
    5: "TRANSFORMER STRESS"
}
Filepath = r"pred_proj/data/MAIIN_DATA (1).csv"
loader = SmartGridDataLoader(Filepath)
loader.load_data()
loader.preprocess()
scaled = loader.scale_data()
dummy_labels =np.zeros(len(scaled))  # Dummy labels for sequence creation
X, _ = loader.create_sequences(scaled, dummy_labels, seq_length=48)  # We only need X for testing   
X = torch.tensor(X, dtype=torch.float32)    
print("input shape:", X.shape  )
autoencoder = LSTMModel(input_size=9) 
autoencoder.load_state_dict(torch.load("C:\\Users\\sharika\\Desktop\\Pred_proj\\pred_proj\\lstm_model.pth", weights_only=True)) #loads trained weights
autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(X)

errors  = torch.mean((X - reconstructed) ** 2, dim=(1, 2)).numpy()
threshold = np.percentile(errors, 95)   
anomalies = errors > threshold
print("\nTotal anomalies detected:", anomalies.sum())
classifier = LSTMClassifier()
classifier.load_state_dict(torch.load("C:\\Users\\sharika\\Desktop\\Pred_proj\\pred_proj\\lstm_classifier.pth", weights_only=True)) #loads trained weights
classifier.eval()
fault_predictions = []
with torch.no_grad():
    outputs = classifier(X)
predicted_labels = torch.argmax(outputs, dim=1).numpy()
for i in range(len(predicted_labels)):
    if anomalies[i]:  # Only print faults for detected anomalies
        fault_predictions.append(FAULTS[predicted_labels[i]])   
    else : 
        fault_predictions.append("Normal")  # No Fault for non-anomalies

print("\nFAULTS DETECTED:\n")

for i in range(len(fault_predictions)):

    if anomalies[i]:

        print(
            f"Sequence {i} → "
            f"{fault_predictions[i]}"
        )

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(16,6))

plt.plot(
    errors,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    color='red',
    linestyle='--',
    label="Threshold"
)

# ==========================================
# MARK ANOMALIES
# ==========================================

anomaly_indices = np.where(anomalies)[0]

plt.scatter(
    anomaly_indices,
    errors[anomaly_indices],
    color='red',
    s=60,
    label="Anomalies"
)

# ==========================================
# ADD FAULT LABELS
# ==========================================

for idx in anomaly_indices:

    plt.text(
        idx,
        errors[idx],
        fault_predictions[idx],
        fontsize=8,
        rotation=45
    )

plt.title(
    "Anomaly Detection + Fault Classification"
)

plt.xlabel("Sequence")

plt.ylabel("Reconstruction Error")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()