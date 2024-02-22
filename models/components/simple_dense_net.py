from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        bias: bool = True,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size, bias=bias),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size, bias=bias),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size, bias=bias),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size, bias=bias),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
