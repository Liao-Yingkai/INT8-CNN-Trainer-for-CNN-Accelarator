"""Official PyTorch eager-mode QAT configuration and checkpoint helpers."""

from pathlib import Path
from typing import Any, Dict

import torch
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, PerChannelMinMaxObserver, QConfig

from model import MnistCnn


def create_fpga_qconfig() -> QConfig:
    """Create QAT configuration: symmetric INT8 activations and per-channel INT8 weights."""
    activation = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    weight = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )
    return QConfig(activation=activation, weight=weight)


def prepare_qat_model(float_checkpoint: str | Path, device: torch.device) -> MnistCnn:
    """Load an FP32 checkpoint and prepare it for fine-tuning with official PyTorch QAT."""
    checkpoint = torch.load(float_checkpoint, map_location="cpu", weights_only=False)
    model = MnistCnn()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fuse_model()
    model.qconfig = create_fpga_qconfig()  # type: ignore
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model.to(device)


def load_qat_model(qat_checkpoint: str | Path, device: torch.device) -> MnistCnn:
    """Rebuild a prepared QAT model and restore its fake-quant parameters and weights."""
    checkpoint: Dict[str, Any] = torch.load(qat_checkpoint, map_location="cpu", weights_only=False)
    model = MnistCnn()
    model.fuse_model()
    model.qconfig = create_fpga_qconfig()  # type: ignore
    torch.ao.quantization.prepare_qat(model, inplace=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def save_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int, accuracy: float) -> None:
    """Save a portable training checkpoint with the model state and accuracy metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "accuracy": accuracy}, path)
