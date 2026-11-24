"""Trace math and portable entry-point regression without VOC or checkpoints."""
from pathlib import Path
import ast
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]/'YOLOV2_TINY'
sys.path.insert(0, str(ROOT))
from dump_trace import (integer_mac, max_pool_hwc, pack_im2row, select_test_indices,
                        stage_names, trace_path, unfold_hwc, write_trace_binary, check_output_paths)
from train import load_config, select_device


class TraceTests(unittest.TestCase):
    """Use independent coordinate and integer references for numerical checks."""

    @classmethod
    def setUpClass(cls):
        """Limit the synthetic CPU workload."""
        torch.set_num_threads(4)

    def test_config_and_device(self):
        """Read the default from any cwd; explicit CPU does not require CUDA."""
        self.assertEqual(load_config()['model']['num_classes'], 20)
        self.assertEqual(select_device('cpu'), torch.device('cpu'))
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(select_device(), torch.device('cpu'))
            with self.assertRaises(RuntimeError):
                select_device('cuda:0')
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                self.assertEqual(load_config()['model']['num_classes'], 20)
            finally:
                os.chdir(previous)

    def test_default_float_checkpoint(self):
        """The default QAT initialization follows the FP32 training output, not a local pretrained file."""
        tree = ast.parse((ROOT/'train.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == '--float-checkpoint'):
                default = next(item.value for item in node.keywords if item.arg == 'default')
                self.assertIsInstance(default, ast.BinOp)
                self.assertEqual(default.left.id, 'ROOT')
                self.assertIsInstance(default.op, ast.Div)
                self.assertEqual(default.right.value, 'checkpoints/float_model.pth')
                return
        self.fail('--float-checkpoint argument missing')

    def test_im2row_coordinates(self):
        """Cover G=1/2/128, both kernels and W=13 spatial tails."""
        for channels, kernel in ((3, 3), (16, 3), (1024, 1), (1024, 3)):
            h, w = 2, 13
            feature = (np.arange(h*w*channels) % 251 - 125).astype(np.int8).reshape(h, w, channels)
            padding = 1 if kernel == 3 else 0
            rows = unfold_hwc(feature, kernel, padding)
            packed = pack_im2row(rows, h, w, channels, kernel)
            groups = (channels+7)//8
            for y in range(h):
                for x in range(w):
                    for kh in range(kernel):
                        for kw in range(kernel):
                            sy, sx = y+kh-padding, x+kw-padding
                            expected = feature[sy, sx] if 0 <= sy < h and 0 <= sx < w else np.zeros(channels, dtype=np.int8)
                            q = kh*kernel+kw
                            np.testing.assert_array_equal(rows[y*w+x, q*channels:(q+1)*channels], expected)
                            for group in range(groups):
                                count = min(8, channels-group*8)
                                actual = packed[y, x//4, q*groups+group, x%4]
                                np.testing.assert_array_equal(actual[:count], expected[group*8:group*8+count])
                                self.assertTrue(np.all(actual[count:] == 0))
            self.assertTrue(np.all(packed[:, -1, :, 1:] == 0))

    def test_mac_and_pool(self):
        """Long reduction matches INT64; negative pool boundaries never use zero."""
        rng = np.random.default_rng(1)
        a = rng.integers(-128, 128, (3, 9216), dtype=np.int8)
        b = rng.integers(-128, 128, (9216, 5), dtype=np.int8)
        np.testing.assert_array_equal(integer_mac(a, b), a.astype(np.int64) @ b.astype(np.int64))
        values = -np.arange(1, 31, dtype=np.int8).reshape(3, 5, 2)
        for stride in (1, 2):
            actual = max_pool_hwc(values, stride)
            for y, x in np.ndindex(actual.shape[:2]):
                patch_values = values[y*stride:min(y*stride+2, 3), x*stride:min(x*stride+2, 5)]
                np.testing.assert_array_equal(actual[y, x], patch_values.max(axis=(0, 1)))

    def test_files_and_sampling(self):
        """Preserve 43 stage names, no bias intermediate, little-endian and no overwrite."""
        self.assertEqual(len(stage_names()), 43)
        self.assertFalse(any('bias' in name for name in stage_names()))
        indices = select_test_indices(4952, 3, 1)
        self.assertEqual(indices, select_test_indices(4952, 3, 1))
        self.assertEqual(len(set(indices)), 3)
        with self.assertRaises(ValueError):
            select_test_indices(1, 2, 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = trace_path(root, 'input_int8_hwc', 0)
            write_trace_binary(path, 'input_int8_hwc', np.array([-128, -1, 127], dtype=np.int8))
            self.assertEqual(path.read_bytes(), b'\x80\xff\x7f')
            with self.assertRaises(FileExistsError):
                check_output_paths(root, 1)
            ints = root/'systolic.bin'
            write_trace_binary(ints, 'systolic_int32', np.array([-1, 16909060], dtype=np.int32))
            self.assertEqual(ints.read_bytes(), b'\xff\xff\xff\xff\x04\x03\x02\x01')


if __name__ == '__main__':
    unittest.main()
