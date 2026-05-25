"""
eval_classifier.py
---------------------------------------------------------
Corrected Evaluation Pipeline

FIXES INCLUDED:
1. Autoencoder properly gates classifier
2. Classifier runs ONLY on anomalous sequences
3. Prevents impossible percentages (>100%)
4. Removes NORMAL class from anomaly classification
5. Proper fault overlays for:
      - current faults on current plots
      - voltage faults on voltage plots
      - load faults on load plots
6. Correct fault distribution statistics
7. Cleaner visualization logic
---------------------------------------------------------

Run:
    python eval_classifier.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter

from dataload import (
    SmartGridDataLoader,
    FAULT_CLASSES,
    FAULT_COLORS,
    V_SAG_LIMIT,
    V_SURGE_LIMIT,
    IMBAL_THRESHOLD,
    OVERLOAD_LIMIT
)

from model import LSTMModel
from classifier import LSTMClassifier


# =========================================================
# CONFIG
# =========================================================

FILEPATH = r"C:\Users\sharika\Desktop\Pred\data\MAIIN_DATA (1).csv"

AUTOENCODER_PATH = "lstm_model.pth"
CLASSIFIER_PATH  = "lstm_classifier.pth"

SEQ_LEN = 48

# Autoencoder anomaly threshold percentile
ANOMALY_PCT = 95

FEAT_NAMES = [
    'IR',
    'IY',
    'IB',
    'VRY',
    'VYB',
    'VBR',
    'Active Load',
    'hour',
    'day'
]


# =========================================================
# LOAD DATA
# =========================================================

print("Loading data …")

loader = SmartGridDataLoader(FILEPATH)

loader.load_data()

loader.preprocess()

scaled = loader.scale_data()

labels = loader.generate_fault_labels(scaled)

X_np, _ = loader.create_sequences(
    scaled,
    labels,
    seq_length=SEQ_LEN
)

X = torch.tensor(
    X_np,
    dtype=torch.float32
)

n_seq = len(X)

print(f"Total sequences : {n_seq}")


# =========================================================
# AUTOENCODER
# =========================================================

print("Running autoencoder …")

autoencoder = LSTMModel(
    input_size=X.shape[2]
)

autoencoder.load_state_dict(
    torch.load(
        AUTOENCODER_PATH,
        weights_only=True
    )
)

autoencoder.eval()

with torch.no_grad():

    recon = autoencoder(X)

# Reconstruction error
recon_err = torch.mean(
    (X - recon) ** 2,
    dim=(1, 2)
).numpy()

# Threshold
threshold = np.percentile(
    recon_err,
    ANOMALY_PCT
)

# Anomaly mask
is_anomaly = recon_err > threshold

print(
    f"Threshold : {threshold:.6f}"
    f"  |  Anomalies : {is_anomaly.sum()}"
)


# =========================================================
# CLASSIFIER
# =========================================================

print("Running classifier …")

clf = LSTMClassifier(
    input_size=X.shape[2]
)

clf.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        weights_only=True
    )
)

clf.eval()

# ---------------------------------------------------------
# ONLY anomalous sequences go to classifier
# ---------------------------------------------------------

anomaly_indices = np.where(is_anomaly)[0]

X_anom = X[anomaly_indices]

print(
    f"Classifier input anomalies : {len(X_anom)}"
)

# ---------------------------------------------------------
# Predict ONLY fault classes (1–5)
# Exclude NORMAL class
# ---------------------------------------------------------

CHUNK = 256

all_preds = []

for s in range(0, len(X_anom), CHUNK):

    batch = X_anom[s:s+CHUNK]

    with torch.no_grad():

        logits = clf(batch)

        probs = torch.softmax(
            logits,
            dim=1
        )

        # -------------------------------------------------
        # Remove NORMAL class from predictions
        # Since autoencoder already marked anomaly
        # -------------------------------------------------

        preds = torch.argmax(
            probs[:, 1:],
            dim=1
        ) + 1

    all_preds.append(
        preds.cpu().numpy()
    )

predicted = np.concatenate(all_preds)

# ---------------------------------------------------------
# Create full-size label array
# Non-anomalies remain NORMAL (0)
# ---------------------------------------------------------

anomaly_labels = np.zeros(
    n_seq,
    dtype=int
)

anomaly_labels[anomaly_indices] = predicted

# Used in reconstruction-error plot
anomalyLabels_autoenc = anomaly_labels


# =========================================================
# HELPERS
# =========================================================

seq_idx = np.arange(n_seq)

# Representative timestep
mid_vals = X_np[:, SEQ_LEN // 2, :]

legend_handles = [

    mpatches.Patch(
        color=FAULT_COLORS[c],
        label=f"{c}: {FAULT_CLASSES[c]}"
    )

    for c in range(1, 6)
]


def scatter_faults(
    ax,
    xvals,
    yvals,
    alabs,
    size=20
):

    for cls in range(1, 6):

        mask = alabs == cls

        if mask.any():

            ax.scatter(
                xvals[mask],
                yvals[mask],
                color=FAULT_COLORS[cls],
                s=size,
                zorder=5,
                linewidths=0
            )


# =========================================================
# PLOT 1 — RECONSTRUCTION ERROR
# =========================================================

fig, ax = plt.subplots(
    figsize=(16, 4)
)

ax.plot(
    seq_idx,
    recon_err,
    color='steelblue',
    lw=1,
    label='Reconstruction Error'
)

ax.axhline(
    threshold,
    color='k',
    ls='--',
    lw=1,
    label=f'Threshold ({threshold:.4f})'
)

scatter_faults(
    ax,
    seq_idx,
    recon_err,
    anomalyLabels_autoenc
)

ax.legend(
    handles=[
        ax.lines[0],
        ax.lines[1]
    ] + legend_handles,
    loc='upper left',
    fontsize=8,
    ncol=4
)

ax.set_title(
    "Reconstruction Error — Autoencoder-Gated Fault Classification",
    fontsize=13
)

ax.set_xlabel("Sequence Index")

ax.set_ylabel("MSE")

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "plot1_recon_error.png",
    dpi=150
)

plt.close()


# =========================================================
# PLOT 2 — THREE-PHASE CURRENTS
# =========================================================

# Only current-related faults
current_faults = np.where(
    np.isin(anomaly_labels, [3, 5]),
    anomaly_labels,
    0
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 10),
    sharex=True
)

fig.suptitle(
    "Three-Phase Currents with Current-Related Faults",
    fontsize=13,
    fontweight='bold'
)

for ax, feat_i, name, color in zip(

    axes,

    [0, 1, 2],

    [
        'IR (Phase R)',
        'IY (Phase Y)',
        'IB (Phase B)'
    ],

    [
        'royalblue',
        'darkorange',
        'forestgreen'
    ]
):

    vals = mid_vals[:, feat_i]

    ax.plot(
        seq_idx,
        vals,
        color=color,
        lw=0.9,
        alpha=0.85
    )

    scatter_faults(
        ax,
        seq_idx,
        vals,
        current_faults
    )

    ax.set_ylabel(
        f"{name}\n(scaled 0–1)",
        fontsize=9
    )

    ax.grid(alpha=0.25)

axes[0].legend(
    handles=legend_handles,
    loc='upper right',
    fontsize=8,
    ncol=2
)

axes[-1].set_xlabel(
    "Sequence Index"
)

plt.tight_layout()

plt.savefig(
    "plot2_currents.png",
    dpi=150
)

plt.close()


# =========================================================
# PLOT 3 — LINE VOLTAGES
# =========================================================

# Only voltage-related faults
voltage_faults = np.where(
    np.isin(anomaly_labels, [1, 2]),
    anomaly_labels,
    0
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(16, 10),
    sharex=True
)

fig.suptitle(
    "Line Voltages with Voltage Faults",
    fontsize=13,
    fontweight='bold'
)

for ax, feat_i, name, color in zip(

    axes,

    [3, 4, 5],

    [
        'VRY (R–Y)',
        'VYB (Y–B)',
        'VBR (B–R)'
    ],

    [
        'crimson',
        'darkorange',
        'mediumpurple'
    ]
):

    vals = mid_vals[:, feat_i]

    ax.plot(
        seq_idx,
        vals,
        color=color,
        lw=0.9,
        alpha=0.85
    )

    ax.axhline(
        V_SAG_LIMIT,
        color='red',
        ls=':',
        lw=1.2
    )

    ax.axhline(
        V_SURGE_LIMIT,
        color='orange',
        ls=':',
        lw=1.2
    )

    scatter_faults(
        ax,
        seq_idx,
        vals,
        voltage_faults
    )

    ax.set_ylabel(
        f"{name}\n(scaled 0–1)",
        fontsize=9
    )

    ax.grid(alpha=0.25)

axes[-1].set_xlabel(
    "Sequence Index"
)

plt.tight_layout()

plt.savefig(
    "plot3_voltages.png",
    dpi=150
)

plt.close()


# =========================================================
# PLOT 4 — ACTIVE LOAD
# =========================================================

# Only load-related faults
load_faults = np.where(
    np.isin(anomaly_labels, [4, 5]),
    anomaly_labels,
    0
)

fig, ax = plt.subplots(
    figsize=(16, 4)
)

vals = mid_vals[:, 6]

ax.plot(
    seq_idx,
    vals,
    color='purple',
    lw=0.9,
    alpha=0.9,
    label='Active Load'
)

ax.axhline(
    OVERLOAD_LIMIT,
    color='red',
    ls='--',
    lw=1.2,
    label=f'Overload limit ({OVERLOAD_LIMIT})'
)

scatter_faults(
    ax,
    seq_idx,
    vals,
    load_faults
)

ax.legend(
    handles=[
        ax.lines[0],
        ax.lines[1]
    ] + legend_handles,
    loc='upper right',
    fontsize=8,
    ncol=3
)

ax.set_title(
    "Active Load with Load-Related Faults",
    fontsize=13
)

ax.set_xlabel(
    "Sequence Index"
)

ax.set_ylabel(
    "Active Load (scaled 0–1)"
)

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "plot4_load.png",
    dpi=150
)

plt.close()


# =========================================================
# PLOT 5 — FAULT DISTRIBUTION
# =========================================================

anom_classes = Counter(
    anomaly_labels[anomaly_labels > 0]
)

classes_present = sorted(anom_classes)

labels_pie = [
    FAULT_CLASSES[c]
    for c in classes_present
]

sizes_pie = [
    anom_classes[c]
    for c in classes_present
]

colors_pie = [
    FAULT_COLORS[c]
    for c in classes_present
]

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

fig.suptitle(
    "Fault Distribution (Autoencoder-Gated)",
    fontsize=13,
    fontweight='bold'
)

# Pie chart
ax1.pie(
    sizes_pie,
    labels=labels_pie,
    colors=colors_pie,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.82
)

ax1.set_title(
    "Fault Share"
)

# Bar chart
bars = ax2.bar(
    labels_pie,
    sizes_pie,
    color=colors_pie,
    edgecolor='k',
    linewidth=0.5
)

ax2.bar_label(
    bars,
    padding=3,
    fontsize=9
)

ax2.set_ylabel(
    "Number of anomalous sequences"
)

ax2.set_title(
    "Fault Counts"
)

ax2.tick_params(
    axis='x',
    rotation=25
)

ax2.grid(
    axis='y',
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "plot5_fault_distribution.png",
    dpi=150
)

plt.close()


# =========================================================
# SUMMARY
# =========================================================

print("\n=== EVALUATION SUMMARY ===")

print(f"Total sequences  : {n_seq}")

print(
    f"Autoenc anomalies: "
    f"{is_anomaly.sum()} "
    f"({is_anomaly.mean()*100:.1f}%)"
)

print(
    f"Recon threshold  : {threshold:.6f}"
)

print("\nFault breakdown:")

for cls in sorted(anom_classes):

    pct = (
        anom_classes[cls]
        / len(anomaly_indices)
    ) * 100

    print(
        f"  {FAULT_CLASSES[cls]:22s}: "
        f"{anom_classes[cls]:4d}  "
        f"({pct:.1f}%)"
    )

saved = [
    "plot1_recon_error.png",
    "plot2_currents.png",
    "plot3_voltages.png",
    "plot4_load.png",
    "plot5_fault_distribution.png"
]

print("\nSaved plots:")

for f in saved:
    print(f"  {f}")