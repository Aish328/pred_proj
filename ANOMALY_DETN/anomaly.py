import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self,input_size = 5, hidden_size = 128):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size = hidden_size,
            batch_first=True
        )
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True)
    def forward(self,x):
        encoded , _ = self.encoder(x)

        last_hidden = encoded[:,-1:,:].repeat(1,x.size(1),1)
        decoded , _ = self.decoder(last_hidden)
        return decoded
