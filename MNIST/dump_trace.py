"""Export MNIST test images' INT8/INT32 intermediate tensors for HLS C simulation."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torchvision import datasets, transforms

from golden_model import GENERATED_DIR, load_fixed_layers, load_metadata, run_fixed_model


ROOT = Path(__file__).resolve().parent
TRACE_DIR = GENERATED_DIR / "trace"
MNIST_DATA_DIR = ROOT / "data"


def load_test_dataset() -> datasets.MNIST:
    """Load the MNIST test set with the same preprocessing used during training."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return datasets.MNIST(
        root=str(MNIST_DATA_DIR), train=False, transform=transform, download=True
    )


def select_test_indices(dataset_size: int, num_samples: int, seed: int) -> List[int]:
    """Select unique test-set indices in deterministic pseudo-random order."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if num_samples > dataset_size:
        raise ValueError(
            f"num_samples={num_samples} exceeds the MNIST test-set size ({dataset_size}); "
            "sampling is without replacement."
        )
    rng = np.random.default_rng(seed)
    return [int(index) for index in rng.choice(dataset_size, size=num_samples, replace=False)]


def load_test_sample(test_set: datasets.MNIST, sample_index: int) -> Tuple[np.ndarray, int]:
    """Load one already-preprocessed test sample as a one-image NCHW batch."""
    image, label = test_set[sample_index]
    return image.unsqueeze(0).numpy(), int(label)


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


def write_trace_header(
    path: Path,
    tensors: Dict[str, np.ndarray],
    sequence: int,
    sample_index: int,
    label: int,
) -> None:
    """Write one image's intermediate tensors into a directly includable C header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    include_guard = f"MNIST_TRACE_DATA_{sequence:04d}_H"
    lines = [
        f"#ifndef {include_guard}",
        f"#define {include_guard}",
        "",
        "#include <stdint.h>",
        "",
        f"/* trace sequence={sequence}, MNIST test index={sample_index}, label={label} */",
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
    lines.append(f"#endif  // {include_guard}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_trace_tensors(trace: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Select the input and module-boundary tensors written to each header."""
    return {
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


def validate_trace_tensors(tensors: Dict[str, np.ndarray]) -> None:
    """Validate dtype and integer range for all exported tensors."""
    for name, array in tensors.items():
        expected_dtype = np.dtype(np.int32 if "int32" in name else np.int8)
        validate_trace_tensor(name, array, expected_dtype)


def clear_previous_trace_headers() -> int:
    """Remove only headers generated by this script so a new batch is unambiguous."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    previous = list(TRACE_DIR.glob("trace_data_*.h"))
    legacy_header = TRACE_DIR / "trace_data.h"
    if legacy_header.exists():
        previous.append(legacy_header)
    for path in previous:
        path.unlink()
    return len(previous)


def main() -> None:
    """Export selected random MNIST test images and all FPGA module boundaries."""
    parser = argparse.ArgumentParser(description="Export MNIST test-image traces for HLS C simulation")
    parser.add_argument(
        "-n", "--num-samples", "--num-images", dest="num_samples", type=int, default=1,
        help="number of unique random test images to export (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="random seed for selecting a reproducible test-image sequence (default: 1)",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    model_input = metadata["model_input"]
    input_scale = float(model_input["scale"])
    layers = load_fixed_layers(metadata)
    test_set = load_test_dataset()
    sample_indices = select_test_indices(len(test_set), args.num_samples, args.seed)
    removed_count = clear_previous_trace_headers()
    if removed_count:
        print(f"Removed {removed_count} previous trace header(s).")

    for sequence, sample_index in enumerate(sample_indices):
        input_nchw, label = load_test_sample(test_set, sample_index)
        trace = run_fixed_model(input_nchw, input_scale, layers)
        tensors = build_trace_tensors(trace)
        validate_trace_tensors(tensors)

        output_path = TRACE_DIR / f"trace_data_{sequence:04d}.h"
        write_trace_header(output_path, tensors, sequence, sample_index, label)
        print(
            f"[{sequence + 1}/{len(sample_indices)}] "
            f"MNIST test index={sample_index}, label={label} -> {output_path}"
        )
        if sequence == 0:
            for name, array in tensors.items():
                print(f"  {name:36s} shape={str(array.shape):18s} dtype={array.dtype}")


if __name__ == "__main__":
    main()
