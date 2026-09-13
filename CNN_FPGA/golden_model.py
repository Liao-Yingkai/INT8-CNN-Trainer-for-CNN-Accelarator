"""Golden integer model for validating the exported FPGA inference parameters.

The reference side is the restored PyTorch QAT model.  The FPGA side reads
only integer arrays from ``generated/headers/*.h`` and performs INT8 x INT8
MACs with INT32 accumulators and biases.  No systolic timing is simulated.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from dataset import create_mnist_loaders
from quant import load_qat_model


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
QAT_CHECKPOINT = ROOT / "checkpoints/qat_model.pth"

# The validation workload is intentionally fixed: no command-line arguments.
VALIDATION_SAMPLES = 1000
INFERENCE_BATCH_SIZE = 100
INT8_MIN, INT8_MAX = np.iinfo(np.int8).min, np.iinfo(np.int8).max
INT32_INFO = np.iinfo(np.int32)


@dataclass(frozen=True)
class FixedLayer:
    """All integer tensors and scale metadata needed by one FPGA compute layer."""

    name: str
    original_weight_shape: Tuple[int, ...]
    export_weight_shape: Tuple[int, ...]
    weight: np.ndarray
    bias: np.ndarray
    multiplier: np.ndarray
    shift: np.ndarray
    input_scale: float
    output_scale: float
    output_zero_point: int


@dataclass
class DifferenceStats:
    """Accumulate elementwise integer differences without retaining every batch."""

    elements: int = 0
    exact: int = 0
    within_one_lsb: int = 0
    absolute_error_sum: int = 0
    maximum_error: int = 0

    def update(self, reference: np.ndarray, actual: np.ndarray) -> None:
        """Add one pair of equally shaped INT8 tensors to the statistics."""
        if reference.shape != actual.shape:
            raise ValueError(f"Comparison shape mismatch: QAT {reference.shape}, FPGA {actual.shape}.")
        difference = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
        self.elements += difference.size
        self.exact += int(np.count_nonzero(difference == 0))
        self.within_one_lsb += int(np.count_nonzero(difference <= 1))
        self.absolute_error_sum += int(difference.sum(dtype=np.int64))
        self.maximum_error = max(self.maximum_error, int(difference.max(initial=0)))

    def print_summary(self, name: str) -> None:
        """Print bit-exact, one-LSB, mean-error, and maximum-error metrics."""
        exact_ratio = self.exact / self.elements if self.elements else 0.0
        one_lsb_ratio = self.within_one_lsb / self.elements if self.elements else 0.0
        mean_error = self.absolute_error_sum / self.elements if self.elements else 0.0
        print(
            f"{name:>6}: exact={exact_ratio:8.4%}, <=1 LSB={one_lsb_ratio:8.4%}, "
            f"MAE={mean_error:.6f}, max={self.maximum_error}"
        )


def parse_c_array(header_path: Path, symbol: str, dtype: np.dtype) -> np.ndarray:
    """Parse one signed integer array from an exported C header.

    Reading the header instead of the parallel NPY file verifies the exact data
    that the HLS build will include.
    """
    if not header_path.exists():
        raise FileNotFoundError(f"Missing generated header: {header_path}")
    text = header_path.read_text(encoding="utf-8")
    pattern = rf"\b{re.escape(symbol)}\s*\[\s*\]\s*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Symbol {symbol!r} was not found in {header_path}.")

    values = [int(token) for token in re.findall(r"[-+]?\d+", match.group(1))]
    if not values:
        raise ValueError(f"Symbol {symbol!r} in {header_path} contains no values.")
    limits = np.iinfo(dtype)
    if min(values) < limits.min or max(values) > limits.max:
        raise OverflowError(f"{symbol} contains a value outside {np.dtype(dtype).name} range.")
    return np.asarray(values, dtype=dtype)


def load_metadata() -> Dict[str, Any]:
    """Load and minimally validate the structural/scaling export metadata."""
    metadata_path = GENERATED_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("generated/metadata.json is missing. Run `python export.py` first.")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata: Dict[str, Any] = json.load(file)
    if int(metadata.get("format_version", 0)) < 4:
        raise ValueError(
            "golden_model.py requires metadata format_version >= 4. "
            "Run the updated `python export.py` to regenerate NHWC-compatible FC weights."
        )
    if metadata.get("hardware_feature_layout") != "NHWC":
        raise ValueError("The golden model requires hardware_feature_layout=NHWC.")
    return metadata


def load_fixed_layers(metadata: Dict[str, Any]) -> Dict[str, FixedLayer]:
    """Build Conv1, Conv2, and FC descriptors from metadata and C headers."""
    layer_entries = metadata.get("layers")
    if not isinstance(layer_entries, list):
        raise ValueError("metadata.json does not contain a valid layers list.")

    layers: Dict[str, FixedLayer] = {}
    for raw_entry in layer_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each metadata layer entry must be an object.")
        entry = raw_entry
        name = str(entry["name"])
        files = entry["files"]
        if not isinstance(files, dict):
            raise ValueError(f"{name}: files metadata must be an object.")

        weight_shape = tuple(int(value) for value in entry["export_weight_shape"])
        original_shape = tuple(int(value) for value in entry["original_weight_shape"])
        output_channels = original_shape[0]
        weight = parse_c_array(GENERATED_DIR / str(files["weight_header"]), f"{name}_weight", np.dtype(np.int8))
        bias = parse_c_array(GENERATED_DIR / str(files["bias_header"]), f"{name}_bias", np.dtype(np.int32))
        multiplier = parse_c_array(
            GENERATED_DIR / str(files["multiplier_header"]), f"{name}_multiplier", np.dtype(np.int32)
        )
        shift = parse_c_array(
            GENERATED_DIR / str(files["shift_header"]), f"{name}_shift", np.dtype(np.int32)
        )

        expected_weight_values = int(np.prod(weight_shape))
        if weight.size != expected_weight_values:
            raise ValueError(f"{name}: header has {weight.size} weights, expected {expected_weight_values}.")
        for array_name, array in (("bias", bias), ("multiplier", multiplier), ("shift", shift)):
            if array.size != output_channels:
                raise ValueError(f"{name}: {array_name} has {array.size} values, expected {output_channels}.")
        if np.any(shift < 0):
            raise ValueError(f"{name}: negative shift is incompatible with an arithmetic right-shift datapath.")
        if any(int(value) != 0 for value in entry["weight_zero_point"]):
            raise ValueError(f"{name}: hardware simulation requires symmetric weight zero_point=0.")
        output_zero_points = [int(value) for value in entry["activation_output_zero_point"]]
        if output_zero_points != [0]:
            raise ValueError(f"{name}: hardware simulation requires symmetric activation zero_point=0.")
        if name == "fc":
            if entry.get("source_input_order") != "C,S" or entry.get("export_input_order") != "S,C":
                raise ValueError("fc: expected an offline [C,S]-to-[S,C] weight conversion.")
            input_channels = int(entry.get("input_channels", 0))
            spatial_elements = int(entry.get("spatial_elements", 0))
            if input_channels <= 0 or spatial_elements <= 0:
                raise ValueError("fc: input_channels and spatial_elements must be positive.")
            if input_channels * spatial_elements != original_shape[1]:
                raise ValueError("fc: C*S does not match Linear input_features.")

        layers[name] = FixedLayer(
            name=name,
            original_weight_shape=original_shape,
            export_weight_shape=weight_shape,
            weight=weight.reshape(weight_shape),
            bias=bias,
            multiplier=multiplier,
            shift=shift,
            input_scale=float(entry["activation_input_scale"]),
            output_scale=float(entry["activation_output_scale"]),
            output_zero_point=output_zero_points[0],
        )

    required = {"conv1", "conv2", "fc"}
    if set(layers) != required:
        raise ValueError(f"Expected exactly {sorted(required)}, found {sorted(layers)}.")
    return layers


def quantize_symmetric_int8(values: np.ndarray, scale: float) -> np.ndarray:
    """Quantize FP32 data with the QAT per-tensor symmetric INT8 rule."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid activation scale: {scale}.")
    quantized = np.rint(values.astype(np.float64, copy=False) / scale)
    return np.clip(quantized, INT8_MIN, INT8_MAX).astype(np.int8)


def preprocess_conv_input_nhwc(input_nhwc: np.ndarray, kernel_h: int,
                               kernel_w: int, padding: int) -> Tuple[np.ndarray, int, int]:
    """Create FPGA patch rows in the exact flattened order ``[KH, KW, IC]``.

    Exported convolution weights are ``[KH, KW, IC, OC] -> [K, OC]``.
    Therefore each input patch must be transposed to the matching K order before
    matrix multiplication.  This function is the software equivalent of an FPGA
    line buffer plus its input-lane arrangement.
    """
    if input_nhwc.ndim != 4:
        raise ValueError(f"Convolution input must be NHWC, got shape {input_nhwc.shape}.")
    padded = np.pad(input_nhwc, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_h, kernel_w), axis=(1, 2))
    # sliding_window_view returns [N, OH, OW, IC, KH, KW].
    windows = windows.transpose(0, 1, 2, 4, 5, 3)
    output_h, output_w = windows.shape[1], windows.shape[2]
    matrix = np.ascontiguousarray(windows).reshape(-1, kernel_h * kernel_w * input_nhwc.shape[3])
    return matrix, output_h, output_w


def check_int32_accumulator(name: str, accumulator: np.ndarray) -> np.ndarray:
    """Verify the mathematical MAC result fits the accelerator's INT32 accumulator."""
    minimum = int(accumulator.min(initial=0))
    maximum = int(accumulator.max(initial=0))
    if minimum < INT32_INFO.min or maximum > INT32_INFO.max:
        raise OverflowError(f"{name}: accumulator range [{minimum}, {maximum}] exceeds int32.")
    return accumulator.astype(np.int32)


def requantize_accumulator(accumulator: np.ndarray, layer: FixedLayer,
                           apply_relu: bool) -> np.ndarray:
    """Apply the exported integer ``(acc * multiplier) >> shift`` operation.

    The product is INT64 so a valid INT32 accumulator times an INT32 multiplier
    cannot overflow.  Conv/ReLU fusion is represented by clearing negative
    accumulators before requantization; the result is then saturated to INT8.
    """
    accumulator_int64 = accumulator.astype(np.int64, copy=False)
    if apply_relu:
        accumulator_int64 = np.maximum(accumulator_int64, 0)
    product = accumulator_int64 * layer.multiplier.astype(np.int64)
    requantized = np.right_shift(product, layer.shift.astype(np.int64))
    return np.clip(requantized, INT8_MIN, INT8_MAX).astype(np.int8)


def fixed_conv2d_nhwc(input_nhwc: np.ndarray, layer: FixedLayer) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run one Conv pipeline and return unfold plus module-boundary outputs."""
    if len(layer.original_weight_shape) != 4:
        raise ValueError(f"{layer.name}: expected a four-dimensional convolution weight.")
    output_channels, input_channels, kernel_h, kernel_w = layer.original_weight_shape
    if input_nhwc.shape[3] != input_channels:
        raise ValueError(f"{layer.name}: input has {input_nhwc.shape[3]} channels, expected {input_channels}.")
    input_matrix, output_h, output_w = preprocess_conv_input_nhwc(input_nhwc, kernel_h, kernel_w, padding=1)
    if input_matrix.shape[1] != layer.weight.shape[0] or layer.weight.shape[1] != output_channels:
        raise ValueError(f"{layer.name}: preprocessed input and exported weight matrix are incompatible.")

    systolic_int64 = input_matrix.astype(np.int64) @ layer.weight.astype(np.int64)
    systolic = check_int32_accumulator(f"{layer.name} systolic", systolic_int64)
    postprocess_int64 = systolic.astype(np.int64) + layer.bias.astype(np.int64)
    postprocess = check_int32_accumulator(f"{layer.name} postprocess", postprocess_int64)
    postprocess = np.maximum(postprocess, 0).astype(np.int32)
    output_int8 = requantize_accumulator(postprocess, layer, apply_relu=False)
    batch = input_nhwc.shape[0]
    output_shape = (batch, output_h, output_w, output_channels)
    unfold = input_matrix.reshape(batch, output_h * output_w, input_matrix.shape[1])
    return output_int8.reshape(output_shape), {
        "unfold": unfold,
        "systolic": systolic.reshape(output_shape),
        "postprocess": postprocess.reshape(output_shape),
        "requant": output_int8.reshape(output_shape),
    }


def max_pool2x2_nhwc(input_nhwc: np.ndarray) -> np.ndarray:
    """Apply stride-2 2x2 MaxPool directly to an INT8 NHWC feature map."""
    batch, height, width, channels = input_nhwc.shape
    if height % 2 or width % 2:
        raise ValueError(f"MaxPool input spatial dimensions must be even, got {height}x{width}.")
    blocked = input_nhwc.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return blocked.max(axis=(2, 4))


def fixed_linear(input_matrix: np.ndarray, layer: FixedLayer) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run the exported FC pipeline and return INT32 stages plus INT8 output."""
    if input_matrix.ndim != 2 or input_matrix.shape[1] != layer.weight.shape[0]:
        raise ValueError(f"{layer.name}: FC input {input_matrix.shape} is incompatible with {layer.weight.shape}.")
    systolic_int64 = input_matrix.astype(np.int64) @ layer.weight.astype(np.int64)
    systolic = check_int32_accumulator(f"{layer.name} systolic", systolic_int64)
    postprocess = check_int32_accumulator(
        f"{layer.name} postprocess", systolic.astype(np.int64) + layer.bias.astype(np.int64)
    )
    output_int8 = requantize_accumulator(postprocess, layer, apply_relu=False)
    return output_int8, {
        "systolic": systolic,
        "postprocess": postprocess,
        "requant": output_int8,
    }


def run_fixed_model(input_nchw: np.ndarray, input_scale: float,
                    layers: Dict[str, FixedLayer]) -> Dict[str, np.ndarray]:
    """Run the complete FPGA mathematical pipeline and return all comparison points."""
    input_int8_nchw = quantize_symmetric_int8(input_nchw, input_scale)
    input_int8_nhwc = np.ascontiguousarray(input_int8_nchw.transpose(0, 2, 3, 1))

    conv1, conv1_stages = fixed_conv2d_nhwc(input_int8_nhwc, layers["conv1"])
    pool1 = max_pool2x2_nhwc(conv1)
    conv2, conv2_stages = fixed_conv2d_nhwc(pool1, layers["conv2"])
    pool2 = max_pool2x2_nhwc(conv2)

    # Conv2/Pool2 already emit NHWC.  The exported FC rows were reordered from
    # PyTorch CHW to HWC offline, so hardware can flatten the feature map
    # directly without a runtime NHWC-to-NCHW transpose.
    fc_input = np.ascontiguousarray(pool2).reshape(pool2.shape[0], -1)
    fc, fc_stages = fixed_linear(fc_input, layers["fc"])
    return {
        "input": input_int8_nhwc,
        "conv1": conv1,
        "pool1": pool1,
        "conv2": conv2,
        "pool2": pool2,
        "fc": fc,
        "conv1_unfold": conv1_stages["unfold"],
        "conv1_systolic": conv1_stages["systolic"],
        "conv1_postprocess": conv1_stages["postprocess"],
        "conv1_requant": conv1_stages["requant"],
        "conv2_unfold": conv2_stages["unfold"],
        "conv2_systolic": conv2_stages["systolic"],
        "conv2_postprocess": conv2_stages["postprocess"],
        "conv2_requant": conv2_stages["requant"],
        "fc_systolic": fc_stages["systolic"],
        "fc_postprocess": fc_stages["postprocess"],
        "fc_requant": fc_stages["requant"],
    }


def register_qat_capture_hooks(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """Capture QAT FakeQuant outputs at the same points returned by the fixed model."""
    captured: Dict[str, torch.Tensor] = {}

    def capture(name: str):
        """Create one forward hook that stores a detached CPU tensor."""
        def hook(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach().cpu()
        return hook

    handles = [
        model.quant.register_forward_hook(capture("input")),
        model.conv1.register_forward_hook(capture("conv1")),
        model.pool1.register_forward_hook(capture("pool1")),
        model.conv2.register_forward_hook(capture("conv2")),
        model.pool2.register_forward_hook(capture("pool2")),
        model.fc.register_forward_hook(capture("fc")),
    ]
    return captured, handles


def qat_tensor_to_int8(tensor: torch.Tensor, scale: float, is_nchw: bool) -> np.ndarray:
    """Recover the integer values represented by a QAT fake-quantized tensor."""
    values = quantize_symmetric_int8(tensor.numpy(), scale)
    if is_nchw:
        values = values.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(values)


def validate_exported_shapes(layers: Dict[str, FixedLayer]) -> None:
    """Derive and check each expected HLS matrix shape from its original weight."""
    for name, layer in layers.items():
        if len(layer.original_weight_shape) == 4:
            output_channels, input_channels, kernel_h, kernel_w = layer.original_weight_shape
            expected_shape = (input_channels * kernel_h * kernel_w, output_channels)
        elif len(layer.original_weight_shape) == 2:
            output_features, input_features = layer.original_weight_shape
            expected_shape = (input_features, output_features)
        else:
            raise ValueError(f"{name}: unsupported original weight shape {layer.original_weight_shape}.")
        if layer.export_weight_shape != expected_shape:
            raise ValueError(f"{name}: exported shape {layer.export_weight_shape}, expected {expected_shape}.")
        print(f"{name}: header weight shape={expected_shape}, values={layer.weight.size}")


def main() -> None:
    """Compare the restored QAT model with the header-driven fixed-point model."""
    if not QAT_CHECKPOINT.exists():
        raise FileNotFoundError("checkpoints/qat_model.pth is missing. Run QAT training first.")
    metadata = load_metadata()
    model_input = metadata.get("model_input")
    if not isinstance(model_input, dict):
        raise ValueError("metadata.json does not contain model_input information.")
    if [int(value) for value in model_input["zero_point"]] != [0]:
        raise ValueError("The FPGA golden model requires symmetric model input zero_point=0.")
    input_scale = float(model_input["scale"])
    layers = load_fixed_layers(metadata)
    validate_exported_shapes(layers)

    qat_model = load_qat_model(QAT_CHECKPOINT, torch.device("cpu"))
    qat_model.eval()
    captured, handles = register_qat_capture_hooks(qat_model)
    _, validation_loader = create_mnist_loaders(
        ROOT / "data", batch_size=INFERENCE_BATCH_SIZE, num_workers=0, download=True
    )

    layer_scales = {
        "input": input_scale,
        "conv1": layers["conv1"].output_scale,
        "pool1": layers["conv1"].output_scale,
        "conv2": layers["conv2"].output_scale,
        "pool2": layers["conv2"].output_scale,
        "fc": layers["fc"].output_scale,
    }
    comparison = {name: DifferenceStats() for name in layer_scales}
    qat_correct = fpga_correct = prediction_matches = processed = 0
    logit_absolute_error_sum = 0.0
    logit_elements = 0
    logit_maximum_error = 0.0

    try:
        with torch.no_grad():
            for images, labels in validation_loader:
                remaining = VALIDATION_SAMPLES - processed
                if remaining <= 0:
                    break
                if images.shape[0] > remaining:
                    images, labels = images[:remaining], labels[:remaining]

                captured.clear()
                qat_logits = qat_model(images)
                fixed = run_fixed_model(images.numpy(), input_scale, layers)
                for name, statistics in comparison.items():
                    is_nchw = name != "fc"
                    qat_int8 = qat_tensor_to_int8(captured[name], layer_scales[name], is_nchw=is_nchw)
                    statistics.update(qat_int8, fixed[name])

                fpga_logits = fixed["fc"].astype(np.float64) * layers["fc"].output_scale
                qat_logits_numpy = qat_logits.cpu().numpy().astype(np.float64, copy=False)
                logit_difference = np.abs(qat_logits_numpy - fpga_logits)
                logit_absolute_error_sum += float(logit_difference.sum())
                logit_elements += logit_difference.size
                logit_maximum_error = max(logit_maximum_error, float(logit_difference.max(initial=0.0)))

                labels_numpy = labels.numpy()
                qat_predictions = qat_logits_numpy.argmax(axis=1)
                fpga_predictions = fpga_logits.argmax(axis=1)
                qat_correct += int(np.count_nonzero(qat_predictions == labels_numpy))
                fpga_correct += int(np.count_nonzero(fpga_predictions == labels_numpy))
                prediction_matches += int(np.count_nonzero(qat_predictions == fpga_predictions))
                processed += images.shape[0]
    finally:
        for handle in handles:
            handle.remove()

    if processed == 0:
        raise RuntimeError("No validation samples were processed.")
    print(f"\nValidated samples: {processed}")
    print("Integer layer comparison (QAT vs FPGA):")
    for name in ("input", "conv1", "pool1", "conv2", "pool2", "fc"):
        comparison[name].print_summary(name)
    print("\nFinal output comparison:")
    print(f"  logits MAE:          {logit_absolute_error_sum / logit_elements:.8f}")
    print(f"  logits max error:    {logit_maximum_error:.8f}")
    print(f"  Top-1 agreement:     {prediction_matches / processed:.4%}")
    print(f"  QAT accuracy:        {qat_correct / processed:.4%}")
    print(f"  FPGA accuracy:       {fpga_correct / processed:.4%}")


if __name__ == "__main__":
    main()
