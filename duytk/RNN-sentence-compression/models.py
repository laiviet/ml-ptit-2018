
from torch import nn
import numpy as np
import torch
from torch.autograd import Variable

class RNNCell(nn.Module):
    def __init__(self , input_dim , hidden_dim ):
        super(RNNCell , self).__init__()
        self.Wh = nn.Linear(hidden_dim , hidden_dim)
        self.Wx = nn.Linear(input_dim  , hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x , h):
        h1 = self.Wh(h)
        h2 = self.Wx(x)
        h = self.activation(h1 + h2)
        # h = self.activation(self.Wh(h) + self.Wx(x))
        return h

class LSTMCell(nn.Module):
    def __init__(self , input_dim , hidden_dim ):
        super(LSTMCell , self).__init__()
        self.Wh = nn.Linear(hidden_dim , hidden_dim)
        self.Wx = nn.Linear(input_dim  , hidden_dim)
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x , h, c):
        i = self.sigm (self.Wh(h) + self.Wx(x))
        f = self.sigm (self.Wh(h) + self.Wx(x))
        o = self.sigm (self.Wh(h) + self.Wx(x))
        g = self.tanh (self.Wh(h) + self.Wx(x))
        c = f*c + i*g
        h = o * self.tanh(c)
        return h , c


class RNNModule(nn.Module):
    def init_hidden(self ):
        zero = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim)))).cuda()
        return zero , zero

    def __init__(self , input_dim , hidden_dim , output_dim = 2, lstm = True):
        super(RNNModule , self).__init__()
        self.lstm = lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if lstm:
            # self.cell = nn.LSTM(self.input_dim , self.hidden_dim , num_layers= 1)
            self.cell = LSTMCell(self.input_dim , self.hidden_dim )
        else:
            self.cell = RNNCell(self.input_dim , self.hidden_dim )

        self.activation = nn.Linear(self.hidden_dim , self.output_dim)

    def forward(self, x ,slen):
        """

        :param x: view B,*,D --??--> padding --> L , B, D


        :return:
        """
        self.batch_size = x.shape[0]
        x = x.transpose(0,1)
        # print x.shape
        llen = x.shape[0]
        h , c = self.init_hidden()
        listH = Variable(torch.FloatTensor(np.empty(0))).cuda()
        id = 0
        for xi in x : # xi : (B,D)
            if self.lstm: h , c  = self.cell(xi , h , c)
            else:         h      = self.cell(xi , h)
            listH = torch.cat((listH , h))

        # listH : L x B x h_dim --> B x L x h_dim ---linear--> B x L x ouput_dim
        listH = listH.view((llen * self.batch_size , self.hidden_dim))
        output = self.activation(listH) # lxb , h
        output = output.view(llen , self.batch_size , self.output_dim).transpose(0,1) # B L O
        for ib in range(self.batch_size):
            le = slen[ib]
            for il in range(le + 1, llen):
                output[ib][il] = torch.FloatTensor([0,1])
        output = output.contiguous().view(llen * self.batch_size, self.output_dim)
        return output
