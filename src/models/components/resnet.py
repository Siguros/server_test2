# Imports from PyTorch.
import torch.nn.functional as F
from torch import Tensor, device, manual_seed
from torch import max as torch_max
from torch import nn, no_grad, save


class ResidualBlock(nn.Module):
    """Residual block of a residual network with option for the skip
    connection."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=stride)
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass."""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual block to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlock(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlock(hidden_ch, hidden_ch))
    return layers


class ResNet(nn.Module):
    """Residual network model."""

    def __init__(self, in_ch, hidden_ch, num_layer, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.layer1 = nn.Sequential(
            *concatenate_layer_blocks(hidden_ch, hidden_ch, num_layer, first_layer=True)
        )
        self.layer2 = nn.Sequential(*concatenate_layer_blocks(hidden_ch, hidden_ch * 2, num_layer))
        self.layer3 = nn.Sequential(
            *concatenate_layer_blocks(hidden_ch * 2, hidden_ch * 4, num_layer)
        )
        self.layer4 = nn.Sequential(
            *concatenate_layer_blocks(hidden_ch * 4, hidden_ch * 8, num_layer)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_ch * 8, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
