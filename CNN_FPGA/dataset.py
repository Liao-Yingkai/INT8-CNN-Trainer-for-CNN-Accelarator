"""MNIST data loader helpers."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_mnist_loaders(
    data_dir: str | Path, batch_size: int, num_workers: int, download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create normalized MNIST training and validation loaders in NCHW format."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    root = str(data_dir)
    train_set = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_set = datasets.MNIST(root=root, train=False, transform=transform, download=download)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(test_set, shuffle=False, **common),
    )
