"""
eval_classifier.py
-------------------
Loads the trained LSTMClassifier and plots:
  1. Reconstruction error (from autoencoder) with classified anomaly markers
  2. Per-feature time-series plots (current, voltage, load) with
     colour-coded fault labels overlaid
  3. Fault-type distribution pie chart

Usage:
    python eval_classifier.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

from dataload import SmartGridDataLoader
from model import LSTMModel                       # your existing autoencoder
from classifier import LSTMClassifier, FAULT_CLASSES, FAULT_COLORS

# =====================================================
# CONFIG
# =====================================================

FILEPATH          = r"C:\Users\sharika\Desktop\Pred\data\MAIIN_DATA (1).csv"
AUTOENCODER_PATH  = "lstm_model.pth"
CLASSIFIER_PATH   = "lstm_classifier.pth"
SEQ_LENGTH        = 48
ANOMALY_PERCENTILE = 95          # same threshold as your eval.py

# =====================================================
# LOAD & PREPROCESS DATA
# =====================================================

print("Loading data...")
loader = SmartGridDataLoader(FILEPATH)
loader.load_data()
loader.preprocess()

scaled = loader.scale_data()
labels = loader.generate_fault_labels(scaled)

X_np, _ = loader.create_sequences(scaled, labels, seq_length=SEQ_LENGTH)
X        = torch.tensor(X_np, dtype=torch.float32)

n_seq    = len(X)
print(f"Total sequences : {n_seq}")

# =====================================================
# FEATURE NAMES (must match dataload.py scale_data order)
# =====================================================

FEATURE_NAMES = ['IR', 'IY', 'IB', 'VRY', 'VYB', 'VBR', 'Active Load', 'hour', 'day']
CURRENT_IDX   = [0, 1, 2]          # IR, IY, IB
VOLTAGE_IDX   = [3, 4, 5]          # VRY, VYB, VBR
LOAD_IDX      = [6]                 # Active Load

# =====================================================
# AUTOENCODER — RECONSTRUCTION ERRORS
# =====================================================

print("Running autoencoder...")
autoencoder = LSTMModel(input_size=X.shape[2])
autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
autoencoder.eval()

with torch.no_grad():
    reconstructed = autoencoder(X)

recon_errors = torch.mean((X - reconstructed) ** 2, dim=(1, 2)).numpy()
threshold    = np.percentile(recon_errors, ANOMALY_PERCENTILE)
is_anomaly   = recon_errors > threshold

print(f"Threshold       : {threshold:.6f}")
print(f"Anomalies found : {is_anomaly.sum()}")

# =====================================================
# CLASSIFIER — FAULT LABELS
# =====================================================

print("Running fault classifier...")
classifier = LSTMClassifier(input_size=X.shape[2])
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, weights_only=True))
classifier.eval()

CHUNK = 256          # process in chunks to avoid OOM on large datasets
all_preds  = []
all_probs  = []

for start in range(0, n_seq, CHUNK):
    xb         = X[start:start + CHUNK]
    preds, probs = classifier.predict(xb)
    all_preds.append(preds.numpy())
    all_probs.append(probs.numpy())

predicted_labels = np.concatenate(all_preds)   # (n_seq,)
predicted_probs  = np.concatenate(all_probs)   # (n_seq, 6)

# Anomaly-only labels (non-anomalies shown as Normal)
anomaly_labels = np.where(is_anomaly, predicted_labels, 0)

print("\nFault distribution (anomalous sequences only):")
for cls, cnt in sorted(Counter(predicted_labels[is_anomaly]).items()):
    print(f"  {FAULT_CLASSES[cls]:20s}: {cnt}")

# =====================================================
# HELPER — scatter anomalies coloured by fault type
# =====================================================

def scatter_faults(ax, x_vals, y_vals, labels_arr, size=18, zorder=5):
    for cls in range(1, 6):
        mask = labels_arr == cls
        if mask.any():
            ax.scatter(
                x_vals[mask], y_vals[mask],
                color=FAULT_COLORS[cls],
                label=FAULT_CLASSES[cls],
                s=size, zorder=zorder, linewidths=0
            )

# =====================================================
# SHARED X-AXIS (sequence indices)
# =====================================================

seq_idx = np.arange(n_seq)

# Mid-point value of each sequence for each feature
# shape: (n_seq, 9)
mid_vals = X_np[:, SEQ_LENGTH // 2, :]

# =====================================================
# PLOT 1 — RECONSTRUCTION ERROR WITH FAULT LABELS
# =====================================================

fig1, ax = plt.subplots(figsize=(16, 4))

ax.plot(seq_idx, recon_errors, color='steelblue', linewidth=1, label='Reconstruction Error')
ax.axhline(threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.4f})')

scatter_faults(ax, seq_idx, recon_errors, anomaly_labels)

ax.set_title("LSTM Autoencoder — Reconstruction Error with Fault Classification", fontsize=13)
ax.set_xlabel("Sequence Index")
ax.set_ylabel("MSE")
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eval_recon_error_classified.png", dpi=150)
plt.show()

# =====================================================
# PLOT 2 — CURRENT (IR, IY, IB)
# =====================================================

fig2, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig2.suptitle("Phase Currents with Fault Classification", fontsize=13, fontweight='bold')

for i, (feat_i, name) in enumerate(zip(CURRENT_IDX, ['IR', 'IY', 'IB'])):
    ax   = axes[i]
    vals = mid_vals[:, feat_i]
    ax.plot(seq_idx, vals, color='royalblue', linewidth=0.8, alpha=0.9)
    scatter_faults(ax, seq_idx, vals, anomaly_labels)
    ax.set_ylabel(f"{name} (scaled)", fontsize=9)
    ax.grid(alpha=0.25)

# Single legend for all subplots
handles = [mpatches.Patch(color=FAULT_COLORS[c], label=FAULT_CLASSES[c])
           for c in range(1, 6)]
axes[0].legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
axes[-1].set_xlabel("Sequence Index")

plt.tight_layout()
plt.savefig("eval_currents_classified.png", dpi=150)
plt.show()

# =====================================================
# PLOT 3 — VOLTAGE (VRY, VYB, VBR)
# =====================================================

fig3, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
fig3.suptitle("Line Voltages with Fault Classification", fontsize=13, fontweight='bold')

for i, (feat_i, name) in enumerate(zip(VOLTAGE_IDX, ['VRY', 'VYB', 'VBR'])):
    ax   = axes[i]
    vals = mid_vals[:, feat_i]
    ax.plot(seq_idx, vals, color='darkorange', linewidth=0.8, alpha=0.9)
    scatter_faults(ax, seq_idx, vals, anomaly_labels)
    ax.set_ylabel(f"{name} (scaled)", fontsize=9)
    ax.grid(alpha=0.25)

handles = [mpatches.Patch(color=FAULT_COLORS[c], label=FAULT_CLASSES[c])
           for c in range(1, 6)]
axes[0].legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
axes[-1].set_xlabel("Sequence Index")

plt.tight_layout()
plt.savefig("eval_voltages_classified.png", dpi=150)
plt.show()

# =====================================================
# PLOT 4 — ACTIVE LOAD
# =====================================================

fig4, ax = plt.subplots(figsize=(16, 3.5))
vals = mid_vals[:, LOAD_IDX[0]]
ax.plot(seq_idx, vals, color='purple', linewidth=0.8, alpha=0.9)
scatter_faults(ax, seq_idx, vals, anomaly_labels)

handles = [mpatches.Patch(color=FAULT_COLORS[c], label=FAULT_CLASSES[c])
           for c in range(1, 6)]
ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
ax.set_title("Active Load with Fault Classification", fontsize=13)
ax.set_xlabel("Sequence Index")
ax.set_ylabel("Active Load (scaled)")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("eval_load_classified.png", dpi=150)
plt.show()

# =====================================================
# PLOT 5 — FAULT DISTRIBUTION (PIE)
# =====================================================

anomaly_class_counts = Counter(predicted_labels[is_anomaly])
labels_pie  = [FAULT_CLASSES[c] for c in sorted(anomaly_class_counts)]
sizes_pie   = [anomaly_class_counts[c] for c in sorted(anomaly_class_counts)]
colors_pie  = [FAULT_COLORS[c] for c in sorted(anomaly_class_counts)]

fig5, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes_pie,
    labels=labels_pie,
    colors=colors_pie,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.82
)
for at in autotexts:
    at.set_fontsize(9)

ax.set_title("Fault Type Distribution\n(Anomalous Sequences Only)", fontsize=13)
plt.tight_layout()
plt.savefig("eval_fault_distribution.png", dpi=150)
plt.show()

# =====================================================
# SUMMARY
# =====================================================

print("\n=== EVALUATION SUMMARY ===")
print(f"Total sequences analysed : {n_seq}")
print(f"Anomalies detected       : {is_anomaly.sum()}  ({is_anomaly.mean()*100:.1f}%)")
print(f"Reconstruction threshold : {threshold:.6f}")
print("\nFault breakdown (anomalous sequences):")
for cls in sorted(anomaly_class_counts):
    pct = anomaly_class_counts[cls] / is_anomaly.sum() * 100
    print(f"  {FAULT_CLASSES[cls]:22s}: {anomaly_class_counts[cls]:4d}  ({pct:.1f}%)")

print("\nSaved plots:")
for fname in [
    "eval_recon_error_classified.png",
    "eval_currents_classified.png",
    "eval_voltages_classified.png",
    "eval_load_classified.png",
    "eval_fault_distribution.png"
]:
    print(f"  {fname}")