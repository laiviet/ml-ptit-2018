import numpy as np
import torch
import torch.nn as nn


class RNN_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN_cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.wx = nn.Linear(self.input_dim, self.hidden_dim)
        self.wh = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x, ht=None):
        # init h0 = 0
        if not ht:
            ht = torch.zeros(1, self.hidden_dim)
        x = self.activation(self.wx(x) + self.wh(ht))
        return x


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_cell = RNN_cell(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, h=None):
        outputs = []
        for xi in x:
            h = self.rnn_cell(xi, h)
            y = self.linear(h)
            outputs.append(y)
        outputs = torch.tensor(outputs)
        return outputs


class LSTM_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.wxi = nn.Linear(self.input_dim, self.hidden_dim)
        self.whi = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wxf = nn.Linear(self.input_dim, self.hidden_dim)
        self.whf = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wxo = nn.Linear(self.input_dim, self.hidden_dim)
        self.who = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.wxg = nn.Linear(self.input_dim, self.hidden_dim)
        self.whg = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoi = nn.Sigmoid()

    def forward(self, x, ht=None, ct=None):
        # init h0 = 0
        if not ht:
            ht = torch.zeros(1, self.hidden_dim)
        # init c0 = 1
        if not ct:
            ct = torch.zeros(1, self.hidden_dim) + 1
        i = self.sigmoi(self.wxi(x) + self.whi(ht))
        f = self.sigmoi(self.wxf(x) + self.whf(ht))
        o = self.sigmoi(self.wxo(x) + self.who(ht))
        g = self.sigmoi(self.wxg(x) + self.whg(ht))
        c = f * ct + i * g
        h = o * self.tanh(c)
        return h, c


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm_cell = LSTM_cell(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, h=None, c=None):
        outputs = []
        for xi in x:
            h, c = self.lstm_cell(xi, h, c)
            y = self.linear(h)
            outputs.append(y)
        outputs = torch.tensor(outputs)
        return outputs
