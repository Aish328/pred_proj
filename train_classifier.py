"""
train_classifier.py
--------------------
Trains LSTMClassifier using rule-based labels from generate_fault_labels().
Run:  python train_classifier.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataload import SmartGridDataLoader, FAULT_CLASSES
from classifier  import LSTMClassifier

# ── CONFIG ────────────────────────────────────────────────────────────
FILEPATH   = r"C:\Users\sharika\Desktop\Pred\data\MAIIN_DATA (1).csv"
SEQ_LEN    = 24
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 0.00005
SAVE_PATH  = "lstm_classifier.pth"

# ── DATA ──────────────────────────────────────────────────────────────
print("Loading data …")
loader = SmartGridDataLoader(FILEPATH)
loader.load_data()
loader.preprocess()
scaled = loader.scale_data()
labels = loader.generate_fault_labels(scaled)

print("\nFault label distribution:")
unique, counts = np.unique(labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls}  {FAULT_CLASSES[cls]:22s}: {cnt:6d}  ({cnt/len(labels)*100:.1f}%)")

X, y = loader.create_sequences(scaled, labels, seq_length=SEQ_LEN)
X    = torch.tensor(X, dtype=torch.float32)
y    = torch.tensor(y, dtype=torch.long)
print(f"\nX shape: {X.shape}   y shape: {y.shape}")

# ── SPLIT ─────────────────────────────────────────────────────────────
split   = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ── CLASS WEIGHTS ─────────────────────────────────────────────────────
# compute_class_weight only covers classes present in y_train.
# Force all 6 classes so the weight tensor always has length 6.
NUM_CLASSES = 6
present_classes = np.unique(y_train.numpy())
cw_raw = compute_class_weight(
    class_weight='balanced',
    classes=present_classes,
    y=y_train.numpy()
)
# Fill missing classes with a high weight so they get attention if they appear
cw = np.ones(NUM_CLASSES, dtype=np.float64)
for cls, w in zip(present_classes, cw_raw):
    cw[cls] = w
cw_tensor = torch.tensor(cw, dtype=torch.float32)
print(f"\nClass weights: { {i: round(float(cw[i]),2) for i in range(NUM_CLASSES)} }")

# ── LOADERS ───────────────────────────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── MODEL ─────────────────────────────────────────────────────────────
model     = LSTMClassifier(input_size=X.shape[2])
criterion = nn.CrossEntropyLoss(weight=cw_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# ── TRAINING LOOP ─────────────────────────────────────────────────────
train_losses, val_losses, val_accs = [], [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    avg_train = total / len(train_loader)
    train_losses.append(avg_train)

    model.eval()
    vl, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            vl    += criterion(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            n       += len(yb)
    avg_val = vl / len(val_loader)
    acc     = correct / n * 100
    val_losses.append(avg_val)
    val_accs.append(acc)
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < prev_lr:
        print(f"  ↓ LR reduced to {new_lr:.6f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  Acc: {acc:.1f}%")

print(f"\nBest model → '{SAVE_PATH}'")

# ── FINAL EVAL ────────────────────────────────────────────────────────
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
model.eval()
all_p, all_l = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        all_p.extend(model(xb).argmax(1).numpy())
        all_l.extend(yb.numpy())

print("\n--- Classification Report ---")
all_class_ids = list(range(6))
print(classification_report(
    all_l, all_p,
    labels=all_class_ids,
    target_names=[FAULT_CLASSES[i] for i in all_class_ids],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(all_l, all_p, labels=all_class_ids)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[FAULT_CLASSES[i] for i in all_class_ids],
            yticklabels=[FAULT_CLASSES[i] for i in all_class_ids])
plt.title("Confusion Matrix — LSTM Fault Classifier")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# Training curves
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(train_losses, label='Train'); a1.plot(val_losses, label='Val')
a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(val_accs, color='green'); a2.set_title("Val Accuracy (%)")
a2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Saved: confusion_matrix.png, training_curves.png")
