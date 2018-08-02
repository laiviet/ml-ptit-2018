import torch.nn as nn
import  torch
import numpy as np

class SimpleRNNCell(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        super(SimpleRNNCell,self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.Wx = nn.Linear(hidden_dim, input_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wy = nn.Linear(input_dim, hidden_dim)
        self.activation =  nn.Tanh()
        self.h = self.init_hidden()

    def init_hidden(self):
        h = np.zeros(self.hidden_dim, self.hidden_dim)
        return h

    def forward(self, x):
        self.h = self.Wh.self.h+self.Wx.x
        self.h = self.activation(h)
        y = self.Wy.h
        return self.h, y

class SimpleRNN(nn.Module):

    def __init__(self, hidden_dim, input_dim):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cell = SimpleRNN(hidden_dim, input_dim)

    def forward(self):

        return h, y
