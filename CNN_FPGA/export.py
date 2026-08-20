"""Export QAT INT8 parameters and quantization metadata for a future HLS flow."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from quant import load_qat_model


ROOT = Path(__file__).resolve().parent


def qparams(fake_quant: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read scale and zero-point from an observer-backed PyTorch fake-quant module."""
    scale, zero_point = fake_quant.calculate_qparams()
    return scale.detach().cpu(), zero_point.detach().cpu()


def export_layer(name: str, layer: nn.Module, input_scale: torch.Tensor,
                 output_scale: torch.Tensor, output_zero_point: torch.Tensor,
                 output_dir: Path) -> Dict[str, object]:
    """Quantize one Conv/Linear weight tensor and export INT8 weights plus INT32 bias."""
    weight_scale, weight_zero_point = qparams(layer.weight_fake_quant)
    weight = layer.weight.detach().cpu()
    quantized_weight = torch.quantize_per_channel(
        weight, weight_scale, weight_zero_point.to(torch.int64), axis=0, dtype=torch.qint8
    )
    np.save(output_dir / f"{name}_weight_int8.npy", quantized_weight.int_repr().numpy())

    bias_scale = input_scale.reshape(-1)[0] * weight_scale
    bias_int32 = torch.round(layer.bias.detach().cpu() / bias_scale).to(torch.int32)
    np.save(output_dir / f"{name}_bias_int32.npy", bias_int32.numpy())
    return {
        "weight_file": f"{name}_weight_int8.npy",
        "weight_shape": list(weight.shape),
        "weight_dtype": "int8",
        "weight_scale": weight_scale.tolist(),
        "weight_zero_point": weight_zero_point.tolist(),
        "weight_qscheme": "per_channel_symmetric",
        "weight_channel_axis": 0,
        "bias_file": f"{name}_bias_int32.npy",
        "bias_dtype": "int32",
        "bias_scale": bias_scale.tolist(),
        "input_scale": input_scale.tolist(),
        "output_scale": output_scale.tolist(),
        "output_zero_point": output_zero_point.tolist(),
    }


def main() -> None:
    """Load QAT weights and write `.npy` tensors plus HLS-friendly JSON metadata."""
    qat_path = ROOT / "checkpoints/qat_model.pth"
    if not qat_path.exists():
        raise FileNotFoundError("QAT checkpoint missing. Run `python train.py --mode qat` first.")
    output_dir = ROOT / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_qat_model(qat_path, torch.device("cpu"))
    model.eval()

    input_scale, input_zero_point = qparams(model.quant.activation_post_process)
    conv1_output_scale, conv1_output_zero_point = qparams(model.conv1.activation_post_process)
    conv2_output_scale, conv2_output_zero_point = qparams(model.conv2.activation_post_process)
    fc_output_scale, fc_output_zero_point = qparams(model.fc.activation_post_process)
    metadata = {
        "format_version": 1,
        "layout": "NCHW",
        "activation": {"dtype": "int8", "qscheme": "per_tensor_symmetric", "zero_point": 0},
        "input": {"scale": input_scale.tolist(), "zero_point": input_zero_point.tolist()},
        "layers": {
            "conv1": export_layer("conv1", model.conv1, input_scale, conv1_output_scale, conv1_output_zero_point, output_dir),
            "conv2": export_layer("conv2", model.conv2, conv1_output_scale, conv2_output_scale, conv2_output_zero_point, output_dir),
            "fc": export_layer("fc", model.fc, conv2_output_scale, fc_output_scale, fc_output_zero_point, output_dir),
        },
    }
    with (output_dir / "quantization.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Exported INT8 weights and INT32 biases to {output_dir}")


if __name__ == "__main__":
    main()
