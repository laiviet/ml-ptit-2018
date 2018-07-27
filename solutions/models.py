import torch
import torch.nn as nn


class SimpleFeedForward(nn.Module):
    def __init__(self, layer_dim, activation=nn.Sigmoid()):
        """

        :param layer_dim: list of dims of all layers
        :param activation: an activation function (nn.Sigmoid(), nn.ReLU())
        """
        super(SimpleFeedForward, self).__init__()
        modules = []
        for i in range(0, len(layer_dim) - 1):
            modules.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        """

        :param layer_dim: list of dims of all layers
        :param activation: an activation function (nn.Sigmoid(), nn.ReLU())
        """
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 150, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(150, 300, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(19200, 1024)
        self.last = nn.Linear(1024, 10)

    def forward(self, x):
        #print(x.shape)
        b = x.shape[0]
        c = 3
        h, w = 32, 32

        x = x.view(b, c, h, w)
        #print(x.shape)
        x = self.pooling(self.activation(self.conv1(x)))
        #print(x.shape)
        x = self.pooling(self.activation(self.conv2(x)))

        #print(x.shape)

        x = x.view(b, -1)
        x = self.linear(x)
        result = self.last(x)
        return result
