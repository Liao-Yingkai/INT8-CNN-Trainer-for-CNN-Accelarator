"""Export a QAT MNIST CNN as INT8/INT32 tensors for an HLS CNN accelerator."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from quant import load_qat_model


ROOT = Path(__file__).resolve().parent
INT8_MIN, INT8_MAX = np.iinfo(np.int8).min, np.iinfo(np.int8).max
INT32_INFO = np.iinfo(np.int32)


def extract_fake_quant_params(fake_quant: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Return CPU NumPy scale and zero-point arrays from a QAT FakeQuant module.

    A QAT checkpoint contains FP32 trainable weights.  The attached
    ``weight_fake_quant``/``activation_post_process`` modules contain the
    observer state from which the final quantization parameters are recovered.
    """
    scale, zero_point = fake_quant.calculate_qparams()
    return (
        scale.detach().cpu().numpy().astype(np.float64, copy=False),
        zero_point.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def scalar_activation_scale(scale: np.ndarray, layer_name: str) -> float:
    """Validate and return the one scale required by per-tensor activations."""
    if scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0:
        raise ValueError(f"{layer_name}: expected one finite positive activation scale, got {scale}.")
    return float(scale[0])


def require_symmetric_zero_point(name: str, zero_point: np.ndarray) -> None:
    """Reject unexpected nonzero zero-points unsupported by the target INT8 MAC path."""
    if not np.all(zero_point == 0):
        raise ValueError(f"{name}: expected symmetric quantization with zero_point=0, got {zero_point}.")


def quantize_weight_per_channel(weight: torch.Tensor, weight_scale: np.ndarray,
                                weight_zero_point: np.ndarray) -> np.ndarray:
    """Quantize FP32 Conv/Linear weights to INT8 independently along output channel 0.

    Conv weights are ``[OC, IC, KH, KW]`` and Linear weights are ``[OC, IC]``.
    In both cases the first dimension is the QAT per-channel axis, so each
    output channel uses its own scale: ``round(weight[channel] / scale[channel])``.
    """
    weight_float = weight.detach().cpu().numpy().astype(np.float64, copy=False)
    output_channels = weight_float.shape[0]
    if weight_scale.size != output_channels or weight_zero_point.size != output_channels:
        raise ValueError(
            f"Weight qparams have {weight_scale.size} scales and {weight_zero_point.size} zero-points, "
            f"but tensor has {output_channels} output channels."
        )
    if np.any(weight_scale <= 0) or not np.all(np.isfinite(weight_scale)):
        raise ValueError("Weight scale must be finite and positive.")

    #Broadcast Bias from [OC]/[O_feature] to [OC, 1, 1, 1](CONV) or [O_feature, 1](FC)
    broadcast_shape = (output_channels,) + (1,) * (weight_float.ndim - 1)
    quantized = np.rint(
        weight_float / weight_scale.reshape(broadcast_shape)
        + weight_zero_point.reshape(broadcast_shape)
    )
    return np.clip(quantized, INT8_MIN, INT8_MAX).astype(np.int8)


def convert_bias_to_int32(bias: torch.Tensor, input_scale: float,
                          weight_scale: np.ndarray) -> np.ndarray:
    """Convert FP32 bias to INT32 using ``bias / (input_scale * weight_scale)``.

    One bias value belongs to each output channel, hence the per-channel weight
    scale is used elementwise.  The accelerator adds this INT32 value directly
    to the INT32 MAC accumulator.
    """
    bias_float = bias.detach().cpu().numpy().astype(np.float64, copy=False)
    bias_scale = input_scale * weight_scale
    converted = np.rint(bias_float / bias_scale)
    if np.any(converted < INT32_INFO.min) or np.any(converted > INT32_INFO.max):
        raise OverflowError("Converted bias does not fit in int32.")
    return converted.astype(np.int32)


def _quantize_real_multiplier(real_multiplier: float) -> Tuple[np.int32, np.int32]:
    """Approximate one positive real multiplier as INT32 ``multiplier / 2**shift``.

    ``frexp`` normalizes the real multiplier to a 31-bit signed integer mantissa.
    The HLS datapath should calculate ``(acc * multiplier) >> shift`` using an
    INT64 intermediate product before narrowing to INT8.
    """
    if real_multiplier == 0.0:
        return np.int32(0), np.int32(0)
    if not math.isfinite(real_multiplier) or real_multiplier < 0.0:
        raise ValueError(f"Invalid requantization multiplier: {real_multiplier}.")

    significand, exponent = math.frexp(real_multiplier)
    #Generate 31bit multiplier
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent
    if shift < 0:
        raise ValueError(
            f"Requantization multiplier {real_multiplier} needs a negative right shift. "
            "This exporter only emits the HLS form (acc * multiplier) >> shift."
        )
    return np.int32(multiplier), np.int32(shift)


def calculate_requant_params(input_scale: float, weight_scale: np.ndarray,
                             output_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one INT32 multiplier and shift per output channel.

    The real per-channel conversion factor is
    ``input_scale * weight_scale / output_scale``.  The returned arrays satisfy
    ``real_multiplier ~= multiplier / 2**shift`` and are designed for the HLS
    operation ``acc * multiplier >> shift``.
    """
    real_multiplier = input_scale * weight_scale / output_scale
    parameters = [_quantize_real_multiplier(float(value)) for value in real_multiplier]
    multiplier = np.asarray([value[0] for value in parameters], dtype=np.int32)
    shift = np.asarray([value[1] for value in parameters], dtype=np.int32)
    return multiplier, shift, real_multiplier.astype(np.float64, copy=False)


def convert_conv_weight_layout(weight_int8: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convert Conv weights from PyTorch OIHW to FPGA ``[K, OC]`` in HWC patch order."""
    if weight_int8.ndim != 4:
        raise ValueError(f"Conv weight must be [OC, IC, KH, KW], got shape {weight_int8.shape}.")
    _, input_channels, kernel_h, kernel_w = weight_int8.shape
    exported = weight_int8.transpose(2, 3, 1, 0).reshape(kernel_h * kernel_w * input_channels, -1)
    return np.ascontiguousarray(exported), "[OC,IC,KH,KW] -> [KH,KW,IC,OC] -> [K_HWC,OC]"


def infer_linear_input_shape_chw(output_channels: int,
                                 input_features: int) -> Tuple[int, int, int]:
    """Infer square FC input shape from the final Conv channels and FC feature count."""
    spatial_size, remainder = divmod(input_features, output_channels)
    if remainder != 0:
        raise ValueError(
            f"Linear input_features={input_features} is not divisible by "
            f"the final Conv output_channels={output_channels}."
        )
    spatial_side = math.isqrt(spatial_size)
    if spatial_side * spatial_side != spatial_size:
        raise ValueError(
            f"Cannot infer a square H=W feature map from {spatial_size} spatial values."
        )
    return output_channels, spatial_side, spatial_side


def convert_linear_weight_layout(weight_int8: np.ndarray,
                                 input_shape_chw: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
    """Reorder FC weights so an NHWC feature map can be flattened directly.

    PyTorch trained each FC row against a CHW-flattened feature map.  The FPGA
    convolution pipeline emits NHWC, so the K dimension is permuted offline
    from CHW to HWC while the output-feature dimension remains unchanged.
    """
    if weight_int8.ndim != 2:
        raise ValueError(f"Linear weight must be [OC, K], got shape {weight_int8.shape}.")
    output_features, input_features = weight_int8.shape
    channels, height, width = input_shape_chw
    if channels * height * width != input_features:
        raise ValueError(
            f"Linear input shape {input_shape_chw} contains {channels * height * width} values, "
            f"but weight expects {input_features}."
        )

    exported = (
        weight_int8.reshape(output_features, channels, height, width)
        .transpose(2, 3, 1, 0)
        .reshape(input_features, output_features)
    )
    return np.ascontiguousarray(exported), "[OC,C,H,W] -> [H,W,C,OC] -> [K_HWC,OC]"


def save_npy(path: Path, array: np.ndarray) -> None:
    """Save one contiguous NumPy array and create its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def export_header_file(path: Path, c_type: str, symbol: str, array: np.ndarray) -> None:
    """Write a C header containing one flattened constant tensor for HLS inclusion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array).reshape(-1)
    rows = []
    for start in range(0, values.size, 12):
        row = ", ".join(str(int(value)) for value in values[start:start + 12])
        rows.append(f"    {row}")
    initializer = ",\n".join(rows)
    guard = f"{symbol.upper()}_H"
    path.write_text(
        f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n"
        f"const {c_type} {symbol}[] = {{\n{initializer}\n}};\n\n#endif  // {guard}\n",
        encoding="utf-8",
    )


def export_layer(name: str, layer: nn.Module, input_scale: float,
                 output_scale: float, output_zero_point: np.ndarray,
                 weights_dir: Path, biases_dir: Path, quant_params_dir: Path,
                 headers_dir: Path,
                 linear_input_shape_chw: Tuple[int, int, int] | None = None) -> Dict[str, Any]:
    """Export one Conv/Linear layer and return all HLS-relevant metadata."""
    weight_scale, weight_zero_point = extract_fake_quant_params(layer.weight_fake_quant)
    require_symmetric_zero_point(f"{name} weight", weight_zero_point)
    require_symmetric_zero_point(f"{name} output activation", output_zero_point)

    weight_int8 = quantize_weight_per_channel(layer.weight, weight_scale, weight_zero_point)
    if weight_int8.ndim == 4:
        exported_weight, layout = convert_conv_weight_layout(weight_int8)
    elif weight_int8.ndim == 2 and linear_input_shape_chw is not None:
        exported_weight, layout = convert_linear_weight_layout(weight_int8, linear_input_shape_chw)
    else:
        raise ValueError(f"{name}: Linear export requires linear_input_shape_chw.")
    bias_int32 = convert_bias_to_int32(layer.bias, input_scale, weight_scale)
    multiplier, shift, real_multiplier = calculate_requant_params(input_scale, weight_scale, output_scale)

    save_npy(weights_dir / f"{name}_weight_int8.npy", exported_weight)
    save_npy(biases_dir / f"{name}_bias_int32.npy", bias_int32)
    save_npy(quant_params_dir / f"{name}_weight_scale.npy", weight_scale.astype(np.float32))
    save_npy(quant_params_dir / f"{name}_weight_zero_point.npy", weight_zero_point.astype(np.int32))
    save_npy(quant_params_dir / f"{name}_input_scale.npy", np.asarray([input_scale], dtype=np.float32))
    save_npy(quant_params_dir / f"{name}_output_scale.npy", np.asarray([output_scale], dtype=np.float32))
    save_npy(quant_params_dir / f"{name}_multiplier_int32.npy", multiplier)
    save_npy(quant_params_dir / f"{name}_shift_int32.npy", shift)

    export_header_file(headers_dir / f"{name}_weight.h", "int8_t", f"{name}_weight", exported_weight)
    export_header_file(headers_dir / f"{name}_bias.h", "int32_t", f"{name}_bias", bias_int32)
    export_header_file(headers_dir / f"{name}_quant.h", "int32_t", f"{name}_multiplier", multiplier)
    export_header_file(headers_dir / f"{name}_shift.h", "int32_t", f"{name}_shift", shift)

    layer_metadata: Dict[str, Any] = {
        "name": name,
        "original_weight_shape": list(weight_int8.shape),
        "export_weight_shape": list(exported_weight.shape),
        "weight_dtype": "int8",
        "bias_dtype": "int32",
        "weight_layout": layout,
        "weight_scale": weight_scale.tolist(),
        "weight_zero_point": weight_zero_point.tolist(),
        "activation_input_scale": input_scale,
        "activation_output_scale": output_scale,
        "activation_output_zero_point": output_zero_point.tolist(),
        "bias_scale": (input_scale * weight_scale).tolist(),
        "requant_real_multiplier": real_multiplier.tolist(),
        "requant_multiplier": multiplier.tolist(),
        "requant_shift": shift.tolist(),
        "files": {
            "weight": f"weights/{name}_weight_int8.npy",
            "bias": f"biases/{name}_bias_int32.npy",
            "weight_scale": f"quant_params/{name}_weight_scale.npy",
            "weight_zero_point": f"quant_params/{name}_weight_zero_point.npy",
            "input_scale": f"quant_params/{name}_input_scale.npy",
            "output_scale": f"quant_params/{name}_output_scale.npy",
            "multiplier": f"quant_params/{name}_multiplier_int32.npy",
            "shift": f"quant_params/{name}_shift_int32.npy",
            "weight_header": f"headers/{name}_weight.h",
            "bias_header": f"headers/{name}_bias.h",
            "multiplier_header": f"headers/{name}_quant.h",
            "shift_header": f"headers/{name}_shift.h",
        },
    }
    if linear_input_shape_chw is not None:
        channels, height, width = linear_input_shape_chw
        layer_metadata.update(
            {
                "source_input_layout": "CHW",
                "export_input_layout": "HWC",
                "logical_input_shape_chw": [channels, height, width],
                "logical_input_shape_hwc": [height, width, channels],
            }
        )
    return layer_metadata


def print_layer_summary(metadata: Dict[str, Any]) -> None:
    """Print the requested compact layer-by-layer export validation summary."""
    for layer in metadata["layers"]:
        print(f"{layer['name'].capitalize()}:")
        print(f"  weight original: {layer['original_weight_shape']}")
        print(f"  weight export:   {layer['export_weight_shape']}")
        print(f"  scale count:     {len(layer['weight_scale'])}")
        print(f"  bias count:      {len(layer['bias_scale'])}")
        print("--------------------------------")


def main() -> None:
    """Restore QAT state, export all compute layers, and save metadata.json."""
    qat_path = ROOT / "checkpoints/qat_model.pth"
    if not qat_path.exists():
        raise FileNotFoundError("QAT checkpoint missing. Run `python train.py --mode qat` first.")

    # load_qat_model performs: MnistCnn -> fuse_model -> create qconfig ->
    # prepare_qat -> load_state_dict.  Loading a plain FP32 model is incorrect.
    model = load_qat_model(qat_path, torch.device("cpu"))
    model.eval()

    generated_dir = ROOT / "generated"
    weights_dir = generated_dir / "weights"
    biases_dir = generated_dir / "biases"
    quant_params_dir = generated_dir / "quant_params"
    headers_dir = generated_dir / "headers"

    input_scale_array, input_zero_point = extract_fake_quant_params(model.quant.activation_post_process)
    conv1_output_scale_array, conv1_output_zero_point = extract_fake_quant_params(model.conv1.activation_post_process)
    conv2_output_scale_array, conv2_output_zero_point = extract_fake_quant_params(model.conv2.activation_post_process)
    fc_output_scale_array, fc_output_zero_point = extract_fake_quant_params(model.fc.activation_post_process)
    require_symmetric_zero_point("model input activation", input_zero_point)

    input_scale = scalar_activation_scale(input_scale_array, "model input")
    conv1_output_scale = scalar_activation_scale(conv1_output_scale_array, "conv1 output")
    conv2_output_scale = scalar_activation_scale(conv2_output_scale_array, "conv2 output")
    fc_output_scale = scalar_activation_scale(fc_output_scale_array, "fc output")
    linear_input_shape_chw = infer_linear_input_shape_chw(
        output_channels=model.conv2.out_channels,
        input_features=model.fc.in_features,
    )

    layers = [
        export_layer("conv1", model.conv1, input_scale, conv1_output_scale, conv1_output_zero_point,
                     weights_dir, biases_dir, quant_params_dir, headers_dir),
        export_layer("conv2", model.conv2, conv1_output_scale, conv2_output_scale, conv2_output_zero_point,
                     weights_dir, biases_dir, quant_params_dir, headers_dir),
        export_layer("fc", model.fc, conv2_output_scale, fc_output_scale, fc_output_zero_point,
                     weights_dir, biases_dir, quant_params_dir, headers_dir,
                     linear_input_shape_chw=linear_input_shape_chw),
    ]
    metadata: Dict[str, Any] = {
        "format_version": 3,
        "pytorch_tensor_layout": "NCHW",
        "hardware_feature_layout": "NHWC",
        "hardware_datatypes": {"input": "int8", "weight": "int8", "accumulator": "int32", "bias": "int32"},
        "activation_quantization": {"dtype": "int8", "qscheme": "per_tensor_symmetric", "zero_point": 0},
        "model_input": {"scale": input_scale, "zero_point": input_zero_point.tolist()},
        "layers": layers,
    }
    generated_dir.mkdir(parents=True, exist_ok=True)
    with (generated_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print_layer_summary(metadata)
    print(f"HLS export complete: {generated_dir}")


if __name__ == "__main__":
    main()
