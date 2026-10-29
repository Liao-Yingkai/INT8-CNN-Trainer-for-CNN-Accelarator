"""PyTorch implementation of the canonical YOLOv2-tiny VOC network."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNLeaky(nn.Module):
    """Darknet convolution block with batch normalization and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Build a stride-one, same-padded convolution block."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization, and LeakyReLU."""
        return self.activation(self.bn(self.conv(x)))


class DarknetMaxPool2d(nn.Module):
    """Darknet max-pooling with only the effective right/bottom padding."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        """Set right/bottom padding when stride-one pooling must preserve size."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # The stride-two layers receive even-sized feature maps because the
        # network input dimensions must be divisible by 32. Their trailing
        # Darknet padding is never sampled, so only stride-one pooling needs it.
        self.padding = kernel_size - 1 if stride == 1 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool while padding only the right and bottom edges with negative infinity."""
        pad_before = self.padding // 2
        pad_after = self.padding - pad_before
        if self.padding:
            x = F.pad(
                x,
                (pad_before, pad_after, pad_before, pad_after),
                mode="constant",
                value=float("-inf"),
            )
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class YOLOv2Tiny(nn.Module):
    """YOLOv2-tiny detector producing raw per-anchor prediction logits.

    The convolutional backbone follows the original Darknet YOLOv2-tiny VOC
    configuration. For a 416x416 input and 20 classes, the raw output has
    shape ``[N, 125, 13, 13]`` (5 anchors x (4 box values + objectness + 20
    class values)). Decoding, loss calculation, and NMS are separate concerns.
    """

    def __init__(self, num_classes: int = 20, num_anchors: int = 5) -> None:
        """Create the network for the requested class and anchor counts."""
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive.")

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        prediction_channels = num_anchors * (5 + num_classes)

        self.conv1 = ConvBNLeaky(3, 16, kernel_size=3)
        self.pool1 = DarknetMaxPool2d(2, 2)
        self.conv2 = ConvBNLeaky(16, 32, kernel_size=3)
        self.pool2 = DarknetMaxPool2d(2, 2)
        self.conv3 = ConvBNLeaky(32, 64, kernel_size=3)
        self.pool3 = DarknetMaxPool2d(2, 2)
        self.conv4 = ConvBNLeaky(64, 128, kernel_size=3)
        self.pool4 = DarknetMaxPool2d(2, 2)
        self.conv5 = ConvBNLeaky(128, 256, kernel_size=3)
        self.pool5 = DarknetMaxPool2d(2, 2)
        self.conv6 = ConvBNLeaky(256, 512, kernel_size=3)
        self.pool6 = DarknetMaxPool2d(2, 1)
        self.conv7 = ConvBNLeaky(512, 1024, kernel_size=3)
        self.conv8 = ConvBNLeaky(1024, 1024, kernel_size=3)
        self.prediction = nn.Conv2d(
            1024,
            prediction_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolution and batch-normalization parameters."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                    a=0.1,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw prediction logits in NCHW layout."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected an NCHW RGB batch [N,3,H,W], got {tuple(x.shape)}.")
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise ValueError("Input height and width must both be divisible by 32.")

        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        x = self.pool6(self.conv6(x))
        x = self.conv7(x)
        x = self.conv8(x)
        return self.prediction(x)

    def reshape_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """View NCHW logits as ``[N, anchors, grid_h, grid_w, 5+classes]``."""
        if logits.ndim != 4:
            raise ValueError(f"Expected NCHW logits, got shape {tuple(logits.shape)}.")
        expected_channels = self.num_anchors * (5 + self.num_classes)
        if logits.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} prediction channels, got {logits.shape[1]}."
            )
        batch, _, grid_h, grid_w = logits.shape
        return (
            logits.reshape(batch, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


def build_yolov2_tiny(num_classes: int = 20, num_anchors: int = 5) -> YOLOv2Tiny:
    """Construct a YOLOv2-tiny network with the requested detection head size."""
    return YOLOv2Tiny(num_classes=num_classes, num_anchors=num_anchors)


def expected_output_shape(
    input_size: Tuple[int, int] = (416, 416),
    num_classes: int = 20,
    num_anchors: int = 5,
) -> Tuple[int, int, int]:
    """Return the expected ``[channels, grid_h, grid_w]`` output shape."""
    height, width = input_size
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Input height and width must both be divisible by 32.")
    return num_anchors * (5 + num_classes), height // 32, width // 32
