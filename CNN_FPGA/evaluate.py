"""Evaluate saved FP32 and QAT checkpoints on the MNIST validation set."""

from pathlib import Path

import torch

from dataset import create_mnist_loaders
from model import MnistCnn
from quant import load_qat_model
from train import evaluate_accuracy, load_config, select_device


ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Print FP32 and QAT validation accuracy plus their difference."""
    config = load_config()
    device = select_device()
    _, valid_loader = create_mnist_loaders(ROOT / config["data_dir"], config["batch_size"], config["num_workers"])
    float_path, qat_path = ROOT / "checkpoints/float_model.pth", ROOT / "checkpoints/qat_model.pth"
    if not float_path.exists() or not qat_path.exists():
        raise FileNotFoundError("Both float_model.pth and qat_model.pth are required. Train FP32 then QAT first.")
    float_model = MnistCnn().to(device)
    float_model.load_state_dict(torch.load(float_path, map_location=device, weights_only=False)["model_state_dict"])
    qat_model = load_qat_model(qat_path, device)
    fp32_accuracy = evaluate_accuracy(float_model, valid_loader, device)
    qat_accuracy = evaluate_accuracy(qat_model, valid_loader, device)
    print(f"FP32 accuracy={fp32_accuracy:.4%}")
    print(f"QAT accuracy={qat_accuracy:.4%}")
    print(f"QAT - FP32={qat_accuracy - fp32_accuracy:+.4%}")


if __name__ == "__main__":
    main()
