"""Compatibility shortcut; prefer scripts/inspect_checkpoint.py for new use."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from scripts.inspect_checkpoint import inspect_checkpoint


def main():
    """Keep the former two-best-checkpoint shortcut without printing full tensors."""
    for name in ('float_model_best.pth', 'qat_model_best.pth'):
        inspect_checkpoint(ROOT/'checkpoints'/name)


if __name__ == "__main__":
    main()
