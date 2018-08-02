import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, groups=1) , # 16*16*96
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=3, padding=1, groups=2) , # 16*16*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, groups=2) , # 8*8*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, groups=2),  # 4*4*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2) , # 4*4*256
            nn.MaxPool2d(kernel_size=2) # 2*2*256
        )

        self.FullyConnected = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048*2*2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.view(x.size(0), 2 * 2 * 256)
        x = self.FullyConnected(x)
        return x