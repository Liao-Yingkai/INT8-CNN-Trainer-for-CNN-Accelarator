"""Export one image's INT8/INT32 intermediate tensors for HLS C simulation."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from torchvision import datasets, transforms

from golden_model import GENERATED_DIR, load_fixed_layers, load_metadata, run_fixed_model


ROOT = Path(__file__).resolve().parent
TRACE_DIR = GENERATED_DIR / "trace"
MNIST_DATA_DIR = ROOT / "data"


def load_random_test_image(seed: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """Select one MNIST test image and apply the training preprocessing pipeline."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(
        root=str(MNIST_DATA_DIR), train=False, transform=transform, download=True
    )
    rng = np.random.default_rng(seed)
    sample_index = int(rng.integers(0, len(test_set)))
    image, label = test_set[sample_index]
    return image.unsqueeze(0).numpy(), int(label), sample_index


def validate_trace_tensor(name: str, array: np.ndarray, dtype: np.dtype) -> None:
    """Ensure a trace tensor has the exact integer type and range used by the FPGA."""
    if array.dtype != dtype:
        raise TypeError(f"{name}: expected {dtype}, got {array.dtype}.")
    limits = np.iinfo(dtype)
    if array.size and (array.min() < limits.min or array.max() > limits.max):
        raise OverflowError(f"{name}: values exceed {np.dtype(dtype).name} range.")


def format_c_array(array: np.ndarray) -> str:
    """Format a flattened NumPy array as a readable C initializer."""
    values = array.reshape(-1)
    lines = []
    for start in range(0, values.size, 12):
        lines.append("    " + ", ".join(str(int(value)) for value in values[start:start + 12]))
    return ",\n".join(lines)


def write_trace_header(path: Path, tensors: Dict[str, np.ndarray]) -> None:
    """Write all intermediate tensors into one directly includable C header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#ifndef CNN_FPGA_TRACE_DATA_H",
        "#define CNN_FPGA_TRACE_DATA_H",
        "",
        "#include <stdint.h>",
        "",
    ]
    for name, array in tensors.items():
        symbol = f"trace_{name}"
        count = int(array.size)
        shape = "[" + ", ".join(str(value) for value in array.shape) + "]"
        c_type = "int8_t" if array.dtype == np.int8 else "int32_t"
        lines.extend(
            [
                f"/* {symbol}: shape {shape}, dtype {c_type}, row-major */",
                f"static const {c_type} {symbol}[{count}] = {{",
                format_c_array(array),
                "};",
                "",
            ]
        )
    lines.append("#endif  // CNN_FPGA_TRACE_DATA_H")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Export one random MNIST test image and all FPGA module boundaries."""
    parser = argparse.ArgumentParser(description="Export a random MNIST test image trace")
    parser.add_argument(
        "--seed", type=int, default=1,
        help="random seed for selecting a reproducible test image (default: 1)",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    model_input = metadata["model_input"]
    input_scale = float(model_input["scale"])
    layers = load_fixed_layers(metadata)
    input_nchw, label, sample_index = load_random_test_image(args.seed)
    trace = run_fixed_model(input_nchw, input_scale, layers)

    tensors = {
        "input_int8_hwc": trace["input"][0],
        "conv1_unfold_int8": trace["conv1_unfold"][0],
        "conv1_systolic_int32_nhwc": trace["conv1_systolic"][0],
        "conv1_postprocess_int32_nhwc": trace["conv1_postprocess"][0],
        "conv1_requant_int8_nhwc": trace["conv1_requant"][0],
        "pool1_int8_nhwc": trace["pool1"][0],
        "conv2_unfold_int8": trace["conv2_unfold"][0],
        "conv2_systolic_int32_nhwc": trace["conv2_systolic"][0],
        "conv2_postprocess_int32_nhwc": trace["conv2_postprocess"][0],
        "conv2_requant_int8_nhwc": trace["conv2_requant"][0],
        "pool2_int8_nhwc": trace["pool2"][0],
        "fc_systolic_int32": trace["fc_systolic"][0],
        "fc_postprocess_int32": trace["fc_postprocess"][0],
        "fc_requant_int8": trace["fc_requant"][0],
    }
    for name, array in tensors.items():
        expected_dtype = np.dtype(np.int32 if "int32" in name else np.int8)
        validate_trace_tensor(name, array, expected_dtype)

    output_path = TRACE_DIR / "trace_data.h"
    write_trace_header(output_path, tensors)
    print(f"Selected MNIST test sample index={sample_index}, label={label}")
    print(f"Wrote {len(tensors)} tensors to {output_path}")
    for name, array in tensors.items():
        print(f"  {name:36s} shape={str(array.shape):18s} dtype={array.dtype}")


if __name__ == "__main__":
    main()
