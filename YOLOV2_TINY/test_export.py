"""CPU 轻量回归：只用临时合成 checkpoint，不训练、不下载、不证明检测精度。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.ao.quantization import disable_observer, disable_fake_quant
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from quant import (QAT_SCHEME, build_model, create_fpga_qconfig, load_qat_model,
                   prepare_qat_model, save_checkpoint, validate_qat_structure)
from export import (CHANNELS, LAYER_NAMES, calculate_requant_params, checked_int32,
                    collect_layers, convert_bias_to_int32, convert_conv_weight_layout,
                    export_checkpoint, extract_fake_quant_params, integer_postprocess,
                    pack_weight, positive_scale, quantize_weight_per_channel,
                    validate_hardware_config)

ROOT = Path(__file__).resolve().parent
PACKED_SHAPES = [(4, 9, 4, 8), (8, 18, 4, 8), (16, 36, 4, 8), (32, 72, 4, 8),
                 (64, 144, 4, 8), (128, 288, 4, 8), (256, 576, 4, 8),
                 (256, 1152, 4, 8), (32, 128, 4, 8)]


class ExportTests(unittest.TestCase):
    """使用标准库 unittest，覆盖真实 QAT 结构、文件 ABI 和整数边界。"""

    @classmethod
    def setUpClass(cls):
        """在临时目录初始化、校准合成模型一次；不访问用户训练 checkpoint。"""
        torch.set_num_threads(4)
        torch.manual_seed(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='yolov2_export_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(cls.temp.name)
        cls.config = yaml.safe_load((ROOT/'config.yaml').read_text())
        fp32 = build_model(cls.config).eval()
        cls.images = torch.rand(1, 3, 416, 416)
        with torch.no_grad():
            cls.float_output = fp32(cls.images)
        save_checkpoint(cls.folder/'float.pth', fp32, 0, 0., cls.config)
        del fp32
        cls.model = prepare_qat_model(cls.folder/'float.pth', torch.device('cpu')).eval()
        # 新 QAT 在未启用 FakeQuant 时，前向应等价于 FP32（仅 BN 融合数值误差）。
        cls.model.apply(disable_fake_quant)
        with torch.no_grad():
            folded_output = cls.model(cls.images)
        torch.testing.assert_close(cls.float_output, folded_output, rtol=1e-4, atol=1e-5)
        for module in cls.model.modules():
            if isinstance(module, FakeQuantizeBase):
                module.enable_fake_quant()
        with torch.no_grad():
            cls.model(cls.images)  # 仅随机图的校准冒烟；不是 QAT 训练。
        cls.model.apply(disable_observer)
        save_checkpoint(cls.folder/'qat.pth', cls.model, 0, 0., cls.config)
        cls.metadata = export_checkpoint(cls.folder/'qat.pth', cls.folder/'generated')

    def test_boundaries_and_forward(self):
        """检查 19 个 FQ 的实际执行；Conv1 激活前没有舍入或饱和。"""
        validate_qat_structure(self.model)
        calls, captured, handles = {}, {}, []

        def count(name):
            """记录 FakeQuant 是否确实在 forward 中被调用。"""
            def hook(_module, _inputs, _output):
                calls[name] = calls.get(name, 0) + 1
            return hook

        def capture(name):
            """立即 clone，避免 inplace LeakyReLU 修改激活前的观测值。"""
            def hook(_module, _inputs, output):
                captured[name] = output.detach().clone()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                handles.append(module.register_forward_hook(count(name)))
        handles.extend([self.model.quant.register_forward_hook(capture('input')),
                        self.model.conv1.conv.register_forward_hook(capture('conv')),
                        self.model.conv1.activation.register_forward_hook(capture('activation'))])
        try:
            with torch.no_grad():
                output = self.model(self.images)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.shape, (1, 125, 13, 13))
        self.assertEqual(len(calls), 19)  # 9 weight + 10 activation。
        self.assertTrue(all(value == 1 for value in calls.values()))
        self.assertEqual(sum('weight_fake_quant' in key for key in calls), 9)
        block = self.model.conv1
        with torch.no_grad():
            weight = block.conv.weight_fake_quant(block.conv.weight)
            raw = F.conv2d(captured['input'], weight, block.conv.bias, padding=1)
            expected = block.activation.activation_post_process(F.leaky_relu(raw, .1))
        torch.testing.assert_close(captured['conv'], raw, rtol=0, atol=0)
        torch.testing.assert_close(captured['activation'], expected, rtol=0, atol=0)

    def test_checkpoint_compatibility(self):
        """FP32 不得带 QAT 标记，新 QAT 严格恢复，旧/伪标记 QAT 明确拒绝。"""
        fp = torch.load(self.folder/'float.pth', weights_only=False)
        self.assertNotIn('qat_scheme', fp)
        state = torch.load(self.folder/'qat.pth', weights_only=False)
        self.assertEqual(state['qat_scheme'], QAT_SCHEME)
        restored = load_qat_model(self.folder/'qat.pth', torch.device('cpu'))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        del restored
        state.pop('qat_scheme')
        torch.save(state, self.folder/'old.pth')
        with self.assertRaisesRegex(ValueError, '重新进行 QAT'):
            load_qat_model(self.folder/'old.pth', torch.device('cpu'))
        with self.assertRaises(ValueError):
            prepare_qat_model(self.folder/'qat.pth', torch.device('cpu'))
        fp['qat_scheme'] = QAT_SCHEME
        torch.save(fp, self.folder/'mislabelled.pth')
        with self.assertRaises(RuntimeError):
            load_qat_model(self.folder/'mislabelled.pth', torch.device('cpu'))
        state['qat_scheme'] = QAT_SCHEME
        state['model_state_dict']['conv1.conv.activation_post_process.scale'] = torch.ones(1)
        torch.save(state, self.folder/'extra_old_boundary.pth')
        with self.assertRaises(RuntimeError):
            load_qat_model(self.folder/'extra_old_boundary.pth', torch.device('cpu'))

    def test_all_files_and_layouts(self):
        """九层 BIN 全量解包等于 NPY，并独立按字节索引抽查 Pack/补零/参数。"""
        output = self.folder/'generated'
        meta = json.loads((output/'metadata.json').read_text())
        self.assertEqual((meta['format_version'], meta['weight_layout_version']), (2, 3))
        self.assertEqual(meta['qat_scheme'], QAT_SCHEME)
        self.assertEqual(meta['layer_order'], list(LAYER_NAMES))
        rng = np.random.default_rng(1)
        previous_scale = None
        for index, layer in enumerate(meta['layers']):
            name, files = layer['name'], layer['files']
            cout, cin, kh, kw = layer['original_oihw_shape']
            self.assertEqual((cin, cout), (CHANNELS[index], CHANNELS[index+1]))
            self.assertEqual(tuple(layer['packed_weight_shape']), PACKED_SHAPES[index])
            logical = np.load(output/files['weight']['path'])
            packed = np.fromfile(output/files['weight_bin']['path'], dtype=np.int8).reshape(PACKED_SHAPES[index])
            tiles, tokens, _, _ = packed.shape
            groups = (cin+7)//8
            decoded = packed.reshape(tiles, kh*kw, groups, 4, 8).transpose(1, 2, 4, 0, 3)
            decoded = decoded.reshape(kh*kw, groups*8, tiles*4)
            np.testing.assert_array_equal(decoded[:, :cin, :cout].reshape(-1, cout), logical)
            self.assertTrue(np.all(decoded[:, cin:, :] == 0))
            self.assertTrue(np.all(decoded[:, :, cout:] == 0))
            data = (output/files['weight_bin']['path']).read_bytes()
            self.assertEqual(len(data), tiles*tokens*32)
            for _ in range(20):
                t, token = int(rng.integers(tiles)), int(rng.integers(tokens))
                lane, ilane = int(rng.integers(4)), int(rng.integers(8))
                q, g = divmod(token, groups)
                ic, oc = g*8+ilane, t*4+lane
                expected = int(logical[q*cin+ic, oc]) if ic < cin and oc < cout else 0
                byte = data[32*(t*tokens+token)+8*lane+ilane]
                self.assertEqual(byte, expected & 255)
            bias_data = (output/files['bias_bin']['path']).read_bytes()
            quant_data = (output/files['quant_bin']['path']).read_bytes()
            self.assertEqual(len(bias_data), cout*4)
            self.assertEqual(len(quant_data), 2*cout*4)
            fields = [np.frombuffer(bias_data, dtype='<i4'),
                      *np.frombuffer(quant_data, dtype='<i4').reshape(2, cout)]
            for field_index, key in enumerate(('bias', 'multiplier', 'shift')):
                array = np.load(output/files[key]['path'])
                self.assertEqual(array.dtype, np.dtype('<i4'))
                np.testing.assert_array_equal(fields[field_index], array)
            if previous_scale is not None:
                self.assertEqual(layer['input_scale'], previous_scale)
            previous_scale = layer['output_scale']
            for descriptor in files.values():
                self.assertEqual((output/descriptor['path']).stat().st_size, descriptor['file_size_bytes'])
            self.assertEqual(files['bias_bin']['fields'], ['bias'])
            self.assertEqual(files['bias_bin']['field_byte_offsets'], [0])
            self.assertEqual(files['quant_bin']['fields'], ['multiplier', 'shift'])
            self.assertEqual(files['quant_bin']['field_byte_offsets'], [0, cout*4])
            for key, suffix in (('weight_bin', 'weight'), ('bias_bin', 'bias'), ('quant_bin', 'quant')):
                self.assertEqual(files[key]['path'], f'bins/{name}_{suffix}.bin')
            print(f'PASS {name}: packed={packed.shape}, weight={len(data)} B, '
                  f'bias={len(bias_data)} B, quant={len(quant_data)} B')
        self.assertEqual(meta['layers'][-1]['files']['bias_bin']['file_size_bytes'], 500)
        self.assertEqual(meta['layers'][-1]['files']['quant_bin']['file_size_bytes'], 1000)
        self.assertEqual(len(list(output.glob('bins/*.bin'))), 27)
        self.assertEqual(len(list(output.rglob('*.bin'))), 27)
        self.assertFalse((output/'params').exists())

    def test_signed_bytes_and_group_padding(self):
        """显式覆盖负补码及 Cin=3/16/1024，独立 kernel 补齐，不跨位置拼接。"""
        for cin, cout, kernel in ((3, 5, 3), (16, 8, 3), (1024, 125, 1)):
            weight = (np.arange(cout*cin*kernel*kernel) % 256 - 128).astype(np.int8)
            weight = weight.reshape(cout, cin, kernel, kernel)
            logical = convert_conv_weight_layout(weight)
            packed = pack_weight(logical, weight.shape)
            for oc, ic, kh, kw in ((0, 0, 0, 0), (cout-1, cin-1, kernel-1, kernel-1)):
                token = (kh*kernel+kw)*((cin+7)//8)+ic//8
                self.assertEqual(packed[oc//4, token, oc%4, ic%8], weight[oc, ic, kh, kw])
            self.assertEqual(packed.tobytes()[0], 128)  # -128 补码。
            if cin == 3:
                self.assertTrue(np.all(packed[..., 3:] == 0))
                self.assertTrue(np.all(packed[-1, :, 1:, :] == 0))

    def test_nearest_even_and_q31(self):
        """bias 最近偶数、缺失 bias、Q31 归一化和半整数权重量化。"""
        np.testing.assert_array_equal(convert_bias_to_int32(torch.tensor([.5, 1.5, -.5, -1.5]), 1., np.ones(4)),
                                      [0, 2, 0, -2])
        np.testing.assert_array_equal(convert_bias_to_int32(None, 1., np.ones(3)), [0, 0, 0])
        m, s, r = calculate_requant_params(1., np.array([.1, 1., np.nextafter(1., 0.)]), 1.)
        np.testing.assert_array_equal(m, [1717986918, 1073741824, 1073741824])
        np.testing.assert_array_equal(s, [34, 30, 30])
        np.testing.assert_allclose(np.ldexp(m.astype(float), -s), r, rtol=1e-9)
        fq = create_fpga_qconfig().weight()
        fq(torch.tensor([[[[-127.5, 127.5]]]]))  # symmetric Observer 生成 scale=1。
        fq.disable_observer()
        scale, _ = extract_fake_quant_params(fq, 'test weight', 1)
        self.assertEqual(scale.item(), 1.)
        weight = torch.tensor([[[[-128., -1.5, -.5, .5, 1.5, 128.]]]])
        np.testing.assert_array_equal(quantize_weight_per_channel(weight, scale, fq).reshape(-1),
                                      [-128, -2, 0, 0, 2, 127])

    def test_integer_contract(self):
        """固定 alpha、符号分支、零、饱和、负低位和直接截断必须严格符合公式。"""
        values = np.array([0, 1, -1, -11, 1000, -2000], dtype=np.int32)
        out = integer_postprocess(values, 0, 1 << 30, 30)
        np.testing.assert_array_equal(out, [0, 1, -1, -2, 127, -128])
        # prediction 不启用 alpha；-9>>1=-5，不得变为向零 -4 或最近偶数 -4。
        np.testing.assert_array_equal(integer_postprocess(np.array([-1, -3, 1]), 0, 3, 1, False), [-2, -5, 1])
        self.assertEqual(integer_postprocess(100, -101, 3, 1, False).item(), -2)
        rng = np.random.default_rng(3)
        for leaky in (True, False):
            for _ in range(100):
                acc, bias = int(rng.integers(-10000, 10000)), int(rng.integers(-100, 100))
                m, shift = int(rng.integers(0, 1 << 31)), int(rng.integers(0, 90))
                a = acc+bias
                negative_m = (m*1717986918 + 8589934592)//17179869184
                selected = negative_m if leaky and a < 0 else m
                expected = min(127, max(-128, (a*selected)//(2**shift)))
                self.assertEqual(integer_postprocess(acc, bias, m, shift, leaky).item(), expected)

    def test_invalid_parameters(self):
        """错误输入必须明确失败，不能默认 scale、裁剪溢出参数或悄悄改变 shift。"""
        fq = create_fpga_qconfig().activation()
        with self.assertRaisesRegex(ValueError, '尚未初始化'):
            extract_fake_quant_params(fq, 'uninitialized')
        fq(torch.tensor([-1., 1.]))
        fq.zero_point.fill_(1)
        with self.assertRaisesRegex(ValueError, 'zero_point'):
            extract_fake_quant_params(fq, 'nonzero')
        for value in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                positive_scale(value, 'bad')
            with self.assertRaises(ValueError):
                calculate_requant_params(value, np.ones(1), 1.)
        with self.assertRaises(OverflowError):
            convert_bias_to_int32(torch.tensor([1e30]), 1., np.ones(1))
        with self.assertRaises(OverflowError):
            checked_int32([1 << 31], 'overflow')
        with self.assertRaises(ValueError):
            calculate_requant_params(1., np.array([float(2**40)]), 1.)
        with self.assertRaises(ValueError):
            calculate_requant_params(1e-300, np.array([1e-300]), 1.)
        for args in ((0, 0, 1, -1), (0, 0, -1, 1), (0, 0, 1, 1 << 31)):
            with self.assertRaises((ValueError, OverflowError)):
                integer_postprocess(*args)
        with self.assertRaises(OverflowError):
            integer_postprocess((1 << 31)-1, 1, 1, 0)
        bad = copy.deepcopy(self.config)
        bad['model']['input_height'] = 320
        with self.assertRaisesRegex(ValueError, '416'):
            validate_hardware_config(bad, self.model)

    def test_checkpoint_scale_backend_roundoff(self):
        """允许一 ULP 的后端重算差异，但保留 checkpoint scale；拒绝真实不一致。"""
        fq = create_fpga_qconfig().activation()
        fq(torch.tensor([-1., 1.]))
        fq.disable_observer()
        calculated, _ = fq.calculate_qparams()
        saved = torch.nextafter(calculated, torch.full_like(calculated, float('inf')))
        fq.scale.copy_(saved)
        exported, _ = extract_fake_quant_params(fq, 'one_ulp')
        np.testing.assert_array_equal(exported, saved.numpy().astype(np.float64))
        torch.testing.assert_close(fq.calculate_qparams()[0], calculated, rtol=0, atol=0)
        fq.scale.copy_(torch.nextafter(saved, torch.full_like(saved, float('inf'))))
        with self.assertRaisesRegex(ValueError, '不一致'):
            extract_fake_quant_params(fq, 'two_ulp')

    def test_validation_before_any_write(self):
        """最后一层 bias 出错也必须在创建输出文件之前失败，错误包含层名。"""
        saved = self.model.prediction.bias.detach().clone()
        try:
            with torch.no_grad():
                self.model.prediction.bias.fill_(1e30)
            save_checkpoint(self.folder/'bad.pth', self.model, 0, 0., self.config)
            with self.assertRaisesRegex(ValueError, 'prediction.*bias'):
                export_checkpoint(self.folder/'bad.pth', self.folder/'bad_output')
            self.assertFalse((self.folder/'bad_output').exists())
        finally:
            with torch.no_grad():
                self.model.prediction.bias.copy_(saved)
        legacy = self.folder/'legacy_output'
        legacy.mkdir()
        (legacy/'metadata.json').write_text('{"format_version": 1}')
        with self.assertRaisesRegex(ValueError, '旧版导出'):
            export_checkpoint(self.folder/'qat.pth', legacy)
        self.assertEqual((legacy/'metadata.json').read_text(), '{"format_version": 1}')
        self.assertEqual(len(list(legacy.iterdir())), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
