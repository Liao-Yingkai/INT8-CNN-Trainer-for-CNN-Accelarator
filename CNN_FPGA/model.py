"""MNIST CNN definition using the standard PyTorch NCHW layout."""

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub


class MnistCnn(nn.Module):
    """CNN matching the target FPGA accelerator topology: 1->8->16->10."""

    def __init__(self) -> None:
        """Create all convolution, pooling, and fully connected layers."""
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10, bias=True)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one NCHW batch through the network and return FP32 logits."""
        x = self.quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self) -> None:
        """Fuse Conv+ReLU pairs before QAT, as required by eager-mode quantization."""
        torch.ao.quantization.fuse_modules(self, [["conv1", "relu1"], ["conv2", "relu2"]], inplace=True)
