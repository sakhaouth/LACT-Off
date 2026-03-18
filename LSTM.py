import torch
import torch.nn as nn

class MultiValueLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=6, output_size=12, pred_len = 1):
        super(MultiValueLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.hn = None
        self.cn = None

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize hidden and cell states
        if self.hn == None and self.cn == None:
            self.hn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            self.cn = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward
        out, (hn, cn) = self.lstm(x, (self.hn, self.cn))

        # Take last time step output
        out = out[:, -self.pred_len, :]

        # Fully connected layer
        out = self.fc(out)
        self.hn = hn.detach()
        self.cn = cn.detach()
        return out