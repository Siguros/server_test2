from typing import Optional

import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(
        self,
        num_classes=10,
        in_channels=3,
        kernel_size=3,
        hidden_channels=(64, 128, 256, 512, 512),
        classifier_hidden=4096,
    ):
        h1, h2, h3, h4, h5 = hidden_channels
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, h1, kernel_size=kernel_size, padding=1),  # 64 x 32 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 16 x 16
            nn.Conv2d(h1, h2, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 14 x 14
            nn.Conv2d(h2, h3, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h3, h4, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 x 7 x 7
            nn.Conv2d(h4, h4, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h4, h5, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 x 7 x 7
            nn.Conv2d(h5, h5, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h5, 512, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.LazyLinear(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(classifier_hidden, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(classifier_hidden, num_classes),
        )
        # forward pass to get the output shape
        self.forward(torch.ones((1, in_channels, 32, 32)))

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11avg(nn.Module):
    """VGG11 with average pooling instead of max pooling."""

    def __init__(self, sample_x, **kwargs) -> None:
        super().__init__(**kwargs)
        for i, module in enumerate(self.features.children()):
            # print(i, module)
            # print(sample_x.shape)
            sample_x = module(sample_x)
            if isinstance(module, nn.MaxPool2d):
                self.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)


class Conv2dIm2Col(nn.Module):
    def __init__(self, conv2d, output_shape):
        super().__init__()
        self.unfold = nn.Unfold(
            kernel_size=conv2d.kernel_size, stride=conv2d.stride, padding=conv2d.padding
        )
        # self.fold = nn.Fold(
        #     output_size=output_shape[-2:],
        #     kernel_size=conv2d.kernel_size,
        #     stride=conv2d.stride,
        #     padding=conv2d.padding
        # )
        self.weight = nn.Parameter(conv2d.weight.view(conv2d.out_channels, -1).t())
        self.bias = nn.Parameter(conv2d.bias) if conv2d.bias is not None else None
        self.output_shape = output_shape[-2:]
        del conv2d

    def forward(self, x: torch.Tensor):
        x = self.unfold(x)
        x = x.transpose(1, 2).matmul(self.weight).transpose(1, 2).add(self.bias.view(-1, 1))
        batch, out_channels, _ = x.shape
        x = x.view(batch, out_channels, *self.output_shape)
        # x = x.view(x.shape[0], self.weight.shape[0], x.shape[2], x.shape[3])  # reshape tensor
        # x = self.fold(x)
        return x


class VGG11Im2Col(VGG11avg):
    """Unfold + matmul + fold."""

    def __init__(
        self,
        num_classes=10,
        in_channels=3,
        kernel_size: Optional[int] = 3,
        hidden_channels: tuple | None = None,
        sample_x: torch.Tensor = torch.ones((1, 3, 32, 32)),
        classifier_hidden=4096,
    ):
        if hidden_channels is None:
            self.hidden_channels = tuple(
                (kernel_size**2 * torch.tensor([7, 7 * 2, 7 * 4, 7 * 8, 7 * 9])).tolist()
            )
        else:
            self.hidden_channels = hidden_channels
        super().__init__(
            num_classes, in_channels, kernel_size, self.hidden_channels, classifier_hidden
        )
        for i, module in enumerate(self.features.children()):
            # print(i, module)
            # print(sample_x.shape)
            sample_x = module(sample_x)
            if isinstance(module, nn.Conv2d):
                self.features[i] = Conv2dIm2Col(module, sample_x.shape)

        del sample_x


if __name__ == "__main__":
    _ = VGG11avg()
