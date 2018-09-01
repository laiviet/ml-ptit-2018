from builtins import super
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class SentenceCompression(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_classes, weight_matrix, dictionary_len):
        super(SentenceCompression, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.batch_size = None
        self.embed = nn.Embedding(dictionary_len, input_size)
        self.embed.weight.data = torch.Tensor(weight_matrix)
        self.embed.weight.requires_grad = False
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def init_hc(self):
        h0 = Variable(torch.Tensor(np.zeros((self.batch_size, self.hidden_size)))).cuda()
        c0 = Variable(torch.Tensor(np.zeros((self.batch_size, self.hidden_size)))).cuda()
        return h0, c0

    def forward(self, x, lengths):
        self.batch_size = x.size()[0]
        x = x.transpose(0, 1)
        x = self.embed(x)
        h, c = self.init_hc()
        output = Variable(torch.FloatTensor(np.empty(0))).cuda()
        for xi in x:
            h, c = self.lstm_cell(xi, (h, c))
            output = torch.cat((output, h))

        # output: L*B*D
        output = self.linear(output)
        output = output.view(self.seq_len, self.batch_size, self.num_classes).transpose(0, 1)
        # output: B*L*2
        for i in range(self.batch_size):
            for j in range(lengths[i], self.seq_len):
                output[i][j] = torch.Tensor(np.array([1, 0])).float()

        output = output.contiguous().view(self.batch_size*self.seq_len, self.num_classes)
        return output



