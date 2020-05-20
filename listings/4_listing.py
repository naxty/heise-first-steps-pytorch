import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNNet(nn.Module):
    def __init__(self):
        super(SimpleCNNNet, self).__init__()
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # Dropout Layer: 30% der Neuronen werden zufällig "ausgeschaltet"
        self.dropout = nn.Dropout(p=0.3)
        # Hidden Layer - Anzahl der Neuronen: (I - K + 1)^2 * F / 2*P
        # I = 28: Bild mit 28x28 Pixel
        # K = 3: Kernel Größe
        # F = 16: Anzahl der Filter
        # P = 2: Pooling Operation1
        self.hidden_layer = nn.Linear(2704, 128)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x , (2,2))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x