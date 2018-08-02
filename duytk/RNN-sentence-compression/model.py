
from torch import nn
import numpy as np
import torch
from torch.autograd import Variable

class RNNCell(nn.Module):
    def __init__(self , input_dim , hidden_dim , output_dim):
        super(RNNCell , self).__init__()
        self.Wh = nn.Linear(hidden_dim , hidden_dim)
        self.Wx = nn.Linear(input_dim  , hidden_dim)
        self.Wa = nn.Linear(hidden_dim , output_dim)
        self.activation = nn.Tanh()

    def forward(self, x , h):
        h = self.activation(self.Wh(h) + self.Wx(x))
        y = self.activation(self.Wa(h))
        return h , y

class LSTMCell(nn.Module):
    def __init__(self , input_dim , hidden_dim , output_dim):
        super(LSTMCell , self).__init__()
        self.Wh = nn.Linear(hidden_dim , hidden_dim)
        self.Wx = nn.Linear(input_dim  , hidden_dim)
        self.Wa = nn.Linear(hidden_dim , output_dim)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x , h, c):
        i = self.sigm (self.Wh(h) + self.Wx(x))
        f = self.sigm (self.Wh(h) + self.Wx(x))
        o = self.sigm (self.Wh(h) + self.Wx(x))
        g = self.tanh (self.Wh(h) + self.Wx(x))
        c = f*c + i*g
        h = o * self.tanh(c)
        y = self.sigm(self.Wa(h))
        return h , c , y

class RNNModule(nn.Module):
    def init_hidden(self ):
        zero = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim))))
        return zero , zero

    def __init__(self , batch_size , input_dim , hidden_dim , output_dim = 2, lstm = True):
        super(RNNModule , self).__init__()
        self.lstm = lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        if lstm:
            self.cell = LSTMCell(self.input_dim , self.hidden_dim , self.output_dim)
        else:
            self.cell = RNNCell(self.input_dim , self.hidden_dim , self.output_dim)

    def forward(self, x):
        """

        :param x: view B,L,D -> L,(B,D) .. 0,1,2 -> 1,0,2
        :return: view L,(B,out_dim) -> B , (L , out_dim)
        """
        x = x.transpose(0,1)
        # print x.shape
        llen = x.shape[0]
        h , c = self.init_hidden()
        output = Variable(torch.FloatTensor(np.empty(0)))
        for xi in x :
            # xi : (B,D)
            if self.lstm:
                h , c , y = self.cell(xi , h , c)
            else:
                h , y = self.cell(xi , h)
            output = torch.cat((output , y))

        # output -> Len - Batch - Output_dim --> batch - len - dim
        output = output.view((llen , self.batch_size , self.output_dim)).transpose(0,1)
        return output