from builtins import super
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceCompression(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_classes, weight_matrix, dictionary):
        super(SentenceCompression, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.dictionary = dictionary
        self.embed = nn.Embedding(len(dictionary), input_size)
        self.embed.weight.data.copy_(torch.Tensor(weight_matrix))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)

    def tensor2pad(self, sent_variable):
        sent_len = np.zeros([sent_variable.size()[0]], dtype=np.int)
        features = np.zeros([sent_variable.size()[0], self.input_size], dtype=np.float)
        for i, words in enumerate(sent_variable):
            for j, word in enumerate(words.split()):
                if j >= self.seq_len:
                    break

                if word in self.dictionary:
                    features[i][j] = self.dictionary[word]
                else:
                    features[i][j] = self.dictionary['.']

            sent_len = min(self.seq_len, len(words.split()))

        features = Variable(self.embed(features))
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len = torch.Tensor.long()
        idx_unsort = torch.Tensor(np.argsort(idx_sort)).long()
        idx_sort = torch.Tensor(idx_sort).long()
        features = features.index_select(0, Variable(idx_sort))
        return pack_padded_sequence(features, sent_len, batch_first=True), idx_unsort

    def forward(self, X):
        # batch*seq_len*input_dim
        X, idx_unsort = self.tensor2pad(X)
        # seq_len*batch*input_dim
        X = self.lstm(X)[0]
        # seq_len*batch*hidden_size
        X = pad_packed_sequence(X, batch_first=True)[0]
        X = X.index_select(0, Variable(idx_unsort))
        X = self.linear(X)
        return X

