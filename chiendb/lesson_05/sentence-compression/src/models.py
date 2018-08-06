from builtins import super
import torch.nn as nn


class SentenceCompression(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_classes):
        super(SentenceCompression, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.action = nn.Sigmoid()

    def forward(self, X):
        # batch*seq_len*input_dim
        X = X.transpose((1, 0, 2))
        # seq_len*batch*input_dim
        X = self.lstm(X)
        # seq_len*batch*hidden_size
        X = X.transpose((1, 0, 2))
        X = [self.linear(Xi) for Xi in X]
    
        X = self.linear(X)
        X = self.action(X)
        return X

