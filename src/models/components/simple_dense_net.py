import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        activation: type[nn.Module] = nn.ReLU,
        cfg: list[int] = [784, 128, 128, 10],
        batch_norm: bool = True,
        bias: bool = True,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param cfg: A list of integers representing the number of output features of each layer.
        :param batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        self.activation = activation
        self.model = self.make_layers(cfg, batch_norm, bias)

    def make_layers(self, cfg: list[int], batch_norm: bool, bias: bool) -> nn.Sequential:
        """Create a sequence of linear layers.

        :param cfg: A list of integers representing the number of output features of each layer.
        :param batch_norm: Whether to use batch normalization.
        :return: A sequence of linear layers.
        """
        layers = []
        for i in range(1, len(cfg) - 1):
            layers.append(nn.Linear(cfg[i - 1], cfg[i], bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(cfg[i]))
            layers.append(self.activation())
        layers.append(nn.Linear(cfg[-2], cfg[-1], bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size = x.size(0)

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
