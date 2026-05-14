import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pred_proj.dataload import SmartGridDataLoader
from pred_proj.anomaly import LSTMAutoEncoder
import matplotlib.pyplot as plt
filepath = r"C:\Users\sharika\Desktop\Pred_proj\pred_proj\data\MAIIN_DATA (1).csv"


loader = SmartGridDataLoader(filepath)

loader.load_data()
loader.preprocess()
loader.resample_hourly()

loader.data_hourly['hour'] = loader.data_hourly.index.hour
loader.data_hourly['day'] = loader.data_hourly.index.dayofweek

scaled = loader.scale_data()
X,_ = loader.create_sequences(scaled, seq_length=48)
X = torch.tensor(X,dtype= torch.float32)
dataset = TensorDataset(X)
loader_dl = DataLoader(dataset, batch_size=64, shuffle=True)
model = LSTMAutoEncoder(input_size=X.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#reconstruction
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in loader_dl:
        batch_x = batch[0]
        output = model(batch_x)
        loss = criterion(output,batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} , loss {loss.item():.6f}")
torch.save(model.state_dict(),"anomaly_model.pth")