from torch import nn
import numpy as np
import torch
from torch.autograd import Variable

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

class LSTMnet(nn.Module):
    def __init__(self , input_dim , hidden_dim , output_dim = 2):
        super(LSTMnet , self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cell = LSTMCell(self.input_dim , self.hidden_dim )
        self.linear = nn.Linear(self.hidden_dim , self.output_dim)

    def forward(self, x):
        # B x L x O
        self.batch_size = x.shape[0]
        x = x.transpose(0,1)
        # L x B x O
        sentence_length = x.shape[0]
        h = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim))))
        c = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim))))
        h_list = Variable(torch.FloatTensor(np.empty(0)))

        for xi in x :
            h , c  = self.cell(xi , h , c)
            h_list = torch.cat((h_list , h))

        h_list = h_list.view((sentence_length * self.batch_size , self.hidden_dim))
        output = self.linear(h_list)
        output = output.view(sentence_length , self.batch_size , self.output_dim).transpose(0,1) # B L O
        output = output.contiguous().view(sentence_length * self.batch_size, self.output_dim)

        return output