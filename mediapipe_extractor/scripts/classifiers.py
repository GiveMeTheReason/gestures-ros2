import typing as tp

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 1,
        resample_factor = 2,
        mode = 'identity',
    ) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        adapted_kernel_size = 2 * int(kernel_size * (stride + 1) / 2) - 1
        adapted_stride = stride ** 2
        adapted_dilation = dilation
        adapted_padding = padding * (stride + 1) - dilation * int(1 - stride / 2)
        self.identity_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=adapted_kernel_size,
            stride=adapted_stride,
            dilation=adapted_dilation,
            padding=adapted_padding,
            bias=False,
        )

        if mode == 'identity':
            self.identity_resample = nn.Identity()
        elif mode == 'up':
            self.identity_resample = nn.Upsample(scale_factor=resample_factor, mode='nearest')
        elif mode == 'down':
            self.identity_resample = nn.MaxPool2d(kernel_size=resample_factor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.bn1(tensor)
        out = self.activation(out)
        out = self.identity_resample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        identity = self.identity_resample(tensor)
        identity = self.identity_conv(identity)
        out = out + identity

        return out


class LinearHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 99,
        num_classes: int = 6,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_dim, 48, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(48, num_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(tensor)


class BaselineClassifier(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.linear_head = LinearHead()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.linear_head(tensor)
