import torch
from torch import nn
import torch.nn.functional as F

def train_modality():
    return torch.cuda.is_available()

class CharLSTM(nn.Module):

    def __init__(self, chars, int2char, char2int, train_on_gpu, n_hidden = 256, n_layers = 2, drop_prob = 0.25, lr = 0.001):

        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = chars
        self.int2char = int2char
        self.char2int = char2int

        self.lstm = nn.LSTM( input_size = len(self.chars), hidden_size = n_hidden, num_layers = n_layers,
                             batch_first = True, dropout = drop_prob)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(in_features = n_hidden, out_features = len(self.chars))

    def forward(self, x, hidden):

        lstm_out, hidden = self.lstm(x, hidden)

        out = self.dropout(lstm_out)

        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        return out, hidden

    def init_hidden_state(self, batch_size):

        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
            )

        return hidden
