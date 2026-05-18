import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataload import SmartGridDataLoader
from model import LSTMModel

# 🔹 Load data
filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\MAIIN_DATA (1).csv"

processor = SmartGridDataLoader(filepath)
X = processor.get_processed_data(seq_length=48)
print("\nProcessed Data Shape:")
print(X.shape)
# 🔹 Training setup
criterion = nn.MSELoss()
split = int((0.8)*len(X))
X_train, X_test = X[:split], X[split:]
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = LSTMModel(input_size=X.shape[2])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 50

# 🔹 Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:

        # batch contains only X
        xb = batch[0]

        # forward pass
        reconstructed = model(xb)

        # reconstruction loss
        loss = criterion(
            reconstructed,
            xb
        )

        # backward pass
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f}"
    )

torch.save(model.state_dict(), "lstm_model.pth")
print("model saved as lstm_model.pth")