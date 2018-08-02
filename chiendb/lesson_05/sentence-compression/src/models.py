from builtins import super

import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lx = nn.Linear(input_dim, hidden_dim)
        self.lh = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x, h):
        h = self.activation(self.lx(x) + self.lh(h))
        return h


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = RNNCell(input_dim, hidden_dim)
        self.linear = nn.Linear

    def init_hidden(self):
        pass

    def forward(self, x, h):
        for xi in x:
            h = self.rnn_cell(xi, h)

        pass

