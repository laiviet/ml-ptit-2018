import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class RNNCell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(RNNCell,self).__init__()
        self.Wx = nn.Linear(input_dim,hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wy = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, h , x):
        h = self.activation(self.Wh(h) + self.Wx(x))
        y = self.Wy(h)
        return h , y

class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,batch_size):
        super(RNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.cell = RNNCell(input_dim,hidden_dim)
    def forward(self, X):
        X = X.transpose(0,1)
        list_Y = torch.FloatTensor()
        h = torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim)))
        for xi in X:
            h , y = self.cell( h , xi)
            list_Y = torch.cat((list_Y , y))
        return list_Y

class LSTMCell(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wx = nn.Linear(input_dim,hidden_dim)
        self.Wy = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, h , c , x):
        i = self.sigmoid(self.Wh(h) + self.Wx(x))
        f = self.sigmoid(self.Wh(h) + self.Wx(x))
        o = self.sigmoid(self.Wh(h) + self.Wx(x))
        g = self.activation(self.Wh(h) + self.Wx(x))
        c = f * c + i * g
        h = o * self.activation(c)
        y = self.sigmoid(self.Wy(h))
        return h, c, y

class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,batch_size):
        super(LSTM, self).__init__()
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

    def forward(self, X):
        list_Y = torch.FloatTensor()
        X = X.transpose(0,1)
        h = torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim)))
        c = torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim)))
        for xi in X:
            h ,c ,y = self.cell(h , c , xi)
            list_Y = torch.cat((list_Y , y))
        return list_Y

rnn = RNN(20,10,12)
input = Variable(torch.randn(12,10,20))
output = rnn(input)
print(output)

lstmnet = LSTM(20,10,12)
output = lstmnet(input)
print (output)