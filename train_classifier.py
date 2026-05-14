import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataload import SmartGridDataLoader
from classifier_model import LSTMClassifier

filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\MAIIN_DATA (1).csv"
loader = SmartGridDataLoader(filepath)
loader.load_data()
loader.preprocess()
scaled = loader.scale_data()
labels = loader.generate_fault_labels(scaled)

X,y = loader.create_sequences(scaled, labels, seq_length=48)
print ("\n X shape: ", X.shape)
print ("\n y shape: ", y.shape)

split = int(0.8*len(X))

X_train , y_train = X[:split] , y[:split]
X_test , y_test = X[split:],y[split:]
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)           
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
model = LSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 80
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb  in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "lstm_classifier.pth")
print("\n model saved successfully as lstm_classifier.pth")


