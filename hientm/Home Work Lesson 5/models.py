import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
        if ht is None:
            ht = Variable(torch.FloatTensor(np.zeros((1 , self.hidden_dim))))
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
        # b, l, d
        batch_size = x.shape[0]
        x = x.transpose(0, 1) # l, b, d
        llen = x.shape[0]
        outputs = Variable(torch.FloatTensor(np.empty(0)))
        for xi in x:
            h = self.rnn_cell(xi, h)
            y = self.linear(h)
            outputs = torch.cat((outputs, y))
        outputs = outputs.view((llen, batch_size, self.output_dim)).transpose(0, 1)
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
        # x: b x d
        # init h0 = 0
        if ht is None:
            ht = torch.FloatTensor(np.zeros((x.shape[0] , self.hidden_dim)))
        # init c0 = 1
        if ct is None:
            ct = Variable(torch.FloatTensor(np.zeros((x.shape[0] , self.hidden_dim))) + 1)
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
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h=None
        c=None
        batch_size = x.shape[0]
        length = x.shape[1]
        x = x.transpose(0, 1)  # l, b, d
        outputs = Variable(torch.FloatTensor(np.empty(0)))
        for xi in x:
            h, c = self.lstm_cell(xi, h, c)
            y = self.linear(h)
            outputs = torch.cat((outputs, y))
        outputs = outputs.view((self.output_dim, batch_size, length)).transpose(0, 1)
        return outputs
