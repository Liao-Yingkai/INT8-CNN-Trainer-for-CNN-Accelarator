"""MNIST synthetic regression; no datasets or trained checkpoints required."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]/'MNIST'
sys.path.insert(0, str(ROOT))
from model import MnistCnn
from quant import load_qat_model, prepare_qat_model, save_checkpoint
from export import convert_conv_weight_layout, convert_linear_weight_layout, calculate_requant_params
from train import load_config, select_device


class MnistTests(unittest.TestCase):
    """Keep topology, QAT restoration and HWC export contracts stable."""

    @classmethod
    def setUpClass(cls):
        """Use a small reproducible CPU workload."""
        torch.set_num_threads(4)
        torch.manual_seed(1)

    def test_config_and_device(self):
        """Default configuration is source-relative; CPU remains selectable."""
        self.assertEqual(load_config()['data_dir'], 'data')
        self.assertEqual(select_device('cpu'), torch.device('cpu'))
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(select_device(), torch.device('cpu'))
            with self.assertRaises(RuntimeError):
                select_device('cuda:0')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'custom.yaml'
            path.write_text('data_dir: sample\nbatch_size: 2\n', encoding='utf-8')
            self.assertEqual(load_config(path)['batch_size'], 2)
            previous = Path.cwd()
            try:
                os.chdir(folder)
                self.assertEqual(load_config()['data_dir'], 'data')
            finally:
                os.chdir(previous)

    def test_forward_and_qat_checkpoint(self):
        """Prepare and strictly restore a synthetic QAT checkpoint without training."""
        image = torch.randn(2, 1, 28, 28)
        model = MnistCnn()
        self.assertEqual(model(image).shape, (2, 10))
        with tempfile.TemporaryDirectory() as folder:
            fp, qp = Path(folder)/'float.pth', Path(folder)/'qat.pth'
            save_checkpoint(fp, model, 0, 0.)
            qat = prepare_qat_model(fp, torch.device('cpu'))
            with torch.no_grad():
                qat(image)
            for name, count in (('conv1', 8), ('conv2', 16), ('fc', 10)):
                fake = getattr(qat, name).weight_fake_quant
                self.assertEqual(fake.ch_axis, 0)
                self.assertEqual(fake.scale.numel(), count)
                self.assertTrue(torch.all(fake.zero_point == 0))
            save_checkpoint(qp, qat, 0, 0.)
            restored = load_qat_model(qp, torch.device('cpu'))
            for key, value in qat.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_layouts_and_requant(self):
        """Conv uses HWC patches; FC permutes C,S offline; M/S uses total shift."""
        conv = np.arange(16*8*3*3).astype(np.int8).reshape(16, 8, 3, 3)
        logical, _ = convert_conv_weight_layout(conv)
        np.testing.assert_array_equal(logical, conv.transpose(2, 3, 1, 0).reshape(72, 16))
        fc = np.arange(10*784).astype(np.int8).reshape(10, 784)
        matrix, _ = convert_linear_weight_layout(fc, 16)
        for channel, spatial in ((0, 0), (15, 48), (3, 21)):
            np.testing.assert_array_equal(matrix[spatial*16+channel], fc[:, channel*49+spatial])
        m, shift, ratio = calculate_requant_params(.25, np.array([.5, .25]), .125)
        np.testing.assert_allclose(np.ldexp(m.astype(float), -shift), ratio, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
