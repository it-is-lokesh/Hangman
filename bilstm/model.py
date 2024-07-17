import torch
from torch import nn
from config import config
cfg = config()

class RNN_Unit(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(cfg.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:,-1,:])
        return out

class RNN_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.net = nn.Sequential(RNN_Unit(input_size, hidden_size, num_layers, output_size))

    def forward(self, x):
        return self.net(x)
    
class LSTM_Unit(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(cfg.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(cfg.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])
        return out

class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.net = nn.Sequential(LSTM_Unit(input_size, hidden_size, num_layers, output_size))

    def forward(self, x):
        return self.net(x)
    

class BiLSTM_Unit(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, return_sequences=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(cfg.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(cfg.device)
        out, _ = self.lstm(x, (h0, c0))
        if not self.return_sequences:
            out = out[:,-1,:]
        return out

class BiLSTM_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(28, 128),
            BiLSTM_Unit(128, 128, 3, True),
            BiLSTM_Unit(2*128, 64, 1, True),
            BiLSTM_Unit(2*64, 64, 1, False),
            nn.Dropout(0.75),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.net(x)