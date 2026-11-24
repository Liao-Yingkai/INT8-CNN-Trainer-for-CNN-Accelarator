"""Print checkpoint metadata and tensor shapes without restoring a network."""
import argparse
from pathlib import Path

import torch


def inspect_checkpoint(path):
    """Load a trusted checkpoint on CPU and print a bounded summary."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or 'model_state_dict' not in state:
        raise ValueError('Expected a checkpoint containing model_state_dict.')
    print(f'Checkpoint: {Path(path).name}')
    for name in ('epoch', 'accuracy', 'map50', 'qat_scheme'):
        if name in state:
            print(f'{name}: {state[name]}')
    for name, value in state['model_state_dict'].items():
        if isinstance(value, torch.Tensor):
            print(f'{name}: shape={tuple(value.shape)}, dtype={value.dtype}')


def main():
    """Inspect an explicitly selected file, without assuming trained models exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
