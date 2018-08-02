import numpy as np  
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

class RNNCell(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(RNNCell,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.Tanh()
        self.Wx = nn.Linear(self.input_dim,self.hidden_dim)
        self.Wh = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.Wy = nn.Linear(self.hidden_dim,self.output_dim)
    def forward(self,x,h):
        h = self.activation(self.Wx(x)+self.Wh(h))
        y = self.Wy(h)
        return h,y

class LSTM(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Tanh()
        self.Wix = nn.Linear(self.input_dim,self.output_dim)
        self.Wox = nn.Linear(self.input_dim,self.output_dim)
        self.Wfx = nn.Linear(self.input_dim,self.output_dim)
        self.Wgx = nn.Linear(self.input_dim,self.output_dim)
        self.Wih = nn.Linear(self.output_dim,self.output_dim)
        self.Woh = nn.Linear(self.output_dim,self.output_dim)
        self.Wfh = nn.Linear(self.output_dim,self.output_dim)
        self.Wgh = nn.Linear(self.output_dim,self.output_dim)
    def forward(self,x,h,c):
        i = self.activation1(self.Wix(x)+self.Wih(h))
        f = self.activation1(self.Wfx(x)+self.Wfh(h))
        o = self.activation1(self.Wox(x)+self.Woh(h))
        g = self.activation2(self.Wgx(x)+self.Wgh(h))
        c = f*c+i*g
        h = o*self.activation2(c)
        return h,c,h



class RNN(nn.Module):
    def init_hidden(self,option=1):
        if option==1:
            zero = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.hidden_dim))))
            return zero
        else:
            zero = Variable(torch.FloatTensor(np.zeros((self.batch_size , self.output_dim))))
            return zero,zero

    def __init__(self,batch_size,input_dim,hidden_dim,output_dim,option=1):
        super(RNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        if option==1:
            self.model = RNNCell(self.input_dim,self.hidden_dim,self.output_dim)
        else:
            self.model = LSTM(self.input_dim,self.output_dim)
    def forward(self,x,option=1):
        x = x.transpose(0,1)
        length = x.shape[0]
        if option==1:
            h = self.init_hidden(option)
        else:
            h , c = self.init_hidden(option=2)
        output = Variable(torch.FloatTensor(np.empty(0)))
        for xi in x :
            if option==1:
                h,y = self.model(xi,h)
            else:
                h,c,y = self.model(xi,h,c)
            output = torch.cat((output , y))
        output = output.view((length , self.batch_size , self.output_dim)).transpose(0,1)
        return output
