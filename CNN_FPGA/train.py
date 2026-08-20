"""FP32 training and QAT fine-tuning entry point."""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch import nn, optim

from dataset import create_mnist_loaders
from model import MnistCnn
from quant import prepare_qat_model, save_checkpoint


ROOT = Path(__file__).resolve().parent


def load_config() -> Dict[str, Any]:
    """Read project hyperparameters from config.yaml."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_device() -> torch.device:
    """Select the first CUDA GPU when available, otherwise use CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Calculate classification accuracy without updating model parameters."""
    was_training = model.training
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            correct += (logits.argmax(dim=1) == labels.to(device, non_blocking=True)).sum().item()
            total += labels.size(0)
    model.train(was_training)
    return correct / total


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                valid_loader: torch.utils.data.DataLoader, device: torch.device,
                epochs: int, learning_rate: float, checkpoint_path: Path) -> float:
    """Train or fine-tune a model, print epoch metrics, and save best and final weights."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0.0
    best_path = checkpoint_path.with_name(checkpoint_path.stem + "_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        accuracy = evaluate_accuracy(model, valid_loader, device)
        print(f"epoch={epoch}/{epochs} loss={running_loss / samples:.6f} accuracy={accuracy:.4%}")
        save_checkpoint(checkpoint_path, model, epoch, accuracy)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(best_path, model, epoch, accuracy)
    return best_accuracy


def main() -> None:
    """Parse CLI options and execute either FP32 training or QAT fine-tuning."""
    parser = argparse.ArgumentParser(description="MNIST FP32/QAT training for an INT8 FPGA CNN")
    parser.add_argument("--mode", choices=("float", "qat"), required=True)
    args = parser.parse_args()
    config = load_config()
    device = select_device()
    print(f"Using device: {device}")
    loaders = create_mnist_loaders(ROOT / config["data_dir"], config["batch_size"], config["num_workers"])
    train_loader, valid_loader = loaders

    if args.mode == "float":
        model = MnistCnn().to(device)
        train_model(model, train_loader, valid_loader, device, config["float_epochs"], config["float_learning_rate"], ROOT / "checkpoints/float_model.pth")
        return

    float_path = ROOT / "checkpoints/float_model.pth"
    if not float_path.exists():
        raise FileNotFoundError("Run `python train.py --mode float` before QAT fine-tuning.")
    fp32_model = MnistCnn().to(device)
    fp32_model.load_state_dict(torch.load(float_path, map_location=device, weights_only=False)["model_state_dict"])
    fp32_accuracy = evaluate_accuracy(fp32_model, valid_loader, device)
    qat_model = prepare_qat_model(float_path, device)
    qat_accuracy = train_model(qat_model, train_loader, valid_loader, device, config["qat_epochs"], config["qat_learning_rate"], ROOT / "checkpoints/qat_model.pth")
    print(f"FP32 accuracy={fp32_accuracy:.4%} | QAT accuracy={qat_accuracy:.4%} | delta={qat_accuracy - fp32_accuracy:+.4%}")


if __name__ == "__main__":
    main()
