"""导出单次 PostProcess QAT 参数；本文件也是权重 BIN 与整数数值契约的参考。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization import disable_observer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from model import DEFAULT_VOC_ANCHORS
from quant import QAT_SCHEME, load_qat_model, validate_qat_structure

ROOT = Path(__file__).resolve().parent
ALPHA_MULTIPLIER, ALPHA_SHIFT = 1717986918, 34
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
LAYER_NAMES = tuple(f'conv{i}' for i in range(1, 9)) + ('prediction',)
CHANNELS = (3, 16, 32, 64, 128, 256, 512, 1024, 1024, 125)
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def positive_scale(values, name):
    """拒绝空、非有限或非正 scale，统一以 float64 计算 bias 和实数比例。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array) & (array > 0)):
        raise ValueError(f'{name}: scale 必须已初始化、有限且严格大于 0。')
    return array


def extract_fake_quant_params(fake_quant, name, channels=None):
    """读取官方 Observer 的 qparams，并核实状态已初始化且与当前 FakeQuant 一致。"""
    if not isinstance(fake_quant, FakeQuantizeBase):
        raise ValueError(f'{name}: 缺少官方 FakeQuant。')
    observer = fake_quant.activation_post_process
    minimum, maximum = observer.min_val, observer.max_val
    if (minimum.numel() == 0 or minimum.shape != maximum.shape
            or not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all()
            or not torch.all(minimum <= maximum)):
        raise ValueError(f'{name}: Observer 尚未初始化或 min/max 无效，请先完成新结构 QAT。')
    expected_scheme = torch.per_tensor_symmetric if channels is None else torch.per_channel_symmetric
    if (fake_quant.dtype != torch.qint8 or fake_quant.qscheme != expected_scheme
            or fake_quant.quant_min != -128 or fake_quant.quant_max != 127
            or (channels is not None and fake_quant.ch_axis != 0)):
        raise ValueError(f'{name}: 必须是 [-128,127] signed symmetric INT8，weight 轴为 OC=0。')
    if not bool(fake_quant.fake_quant_enabled.item()):
        raise ValueError(f'{name}: FakeQuant 未启用。')
    # 不前向新图像，不用 max_abs/127 手工估算。冻结后 calculate_qparams 不更新统计。
    scale, zero = fake_quant.calculate_qparams()
    if torch.any(zero != 0) or torch.any(fake_quant.zero_point != 0):
        raise ValueError(f'{name}: 不支持非零 zero_point。')
    if (not torch.equal(scale, fake_quant.scale) or not torch.equal(zero, fake_quant.zero_point)):
        raise ValueError(f'{name}: Observer qparams 与 FakeQuant buffer 不一致。')
    scale = positive_scale(scale.detach().cpu().numpy(), name).reshape(-1)
    zero = zero.detach().cpu().numpy().astype(np.int32).reshape(-1)
    count = 1 if channels is None else channels
    if scale.size != count or zero.size != count or minimum.numel() != count:
        raise ValueError(f'{name}: qparams 数量错误，要求 {count}。')
    if np.any(zero != 0):
        raise ValueError(f'{name}: 不支持非零 zero_point。')
    return scale, zero


def quantize_weight_per_channel(weight, scale, fake_quant):
    """原始 OIHW 的 OC 轴独立量化；最近偶数舍入，并与实际权重 FQ 逐元素核对。"""
    scale = positive_scale(scale, 'weight')
    value = weight.detach().cpu()
    if value.ndim != 4 or scale.shape != (value.shape[0],) or not torch.isfinite(value).all():
        raise ValueError('weight: 要求有限 OIHW 权重和每 OC 一个 scale。')
    # PyTorch CPU FakeQuant 内核用 FP32 reciprocal 乘法实现 weight/scale；
    # 保留相同计算精度，避免改用 float64 除法在半整数附近改变整数权重。
    scales = torch.as_tensor(scale, dtype=value.dtype).reshape(-1, 1, 1, 1)
    quantized = torch.round(value * scales.reciprocal()).clamp(-128, 127).to(torch.int8)
    with torch.no_grad():
        represented = fake_quant(weight.detach()).cpu()
    reference = torch.round(represented.double() / scales.double()).clamp(-128, 127).to(torch.int8)
    if not torch.equal(quantized, reference):
        raise ValueError('weight: 离线整数与 weight FakeQuant 不逐元素一致，拒绝导出。')
    return quantized.numpy()


def checked_int32(values, name):
    """在转换前检查有限性及 INT32 范围，禁止溢出回绕。"""
    array = np.asarray(values)
    if not np.all(np.isfinite(array)) or np.any(array < INT32_MIN) or np.any(array > INT32_MAX):
        raise OverflowError(f'{name}: 参数超出 signed INT32 范围或非有限。')
    return array.astype('<i4')


def convert_bias_to_int32(bias, input_scale, weight_scale):
    """使用 float64 的 input_scale*weight_scale 逐 OC 缩放和最近偶数舍入。"""
    scale = positive_scale(input_scale, 'bias input') * positive_scale(weight_scale, 'bias weight')
    values = np.zeros(scale.size) if bias is None else bias.detach().cpu().numpy().astype(np.float64)
    if values.shape != scale.shape:
        raise ValueError('bias: 长度必须等于 Cout。')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        converted = np.rint(values / positive_scale(scale, 'bias scale'))
    return checked_int32(converted, 'bias')


def calculate_requant_params(input_scale, weight_scale, output_scale):
    """frexp/Q31 编码 r≈M/2**S；S 已是总右移量，不额外右移 31 位。"""
    ins = positive_scale(input_scale, 'requant input')
    outs = positive_scale(output_scale, 'requant output')
    if ins.size != 1 or outs.size != 1:
        raise ValueError('requant: activation scale 必须是 per-tensor 标量。')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        real = ins.item() * positive_scale(weight_scale, 'requant weight') / outs.item()
    real = positive_scale(real, 'requant ratio').reshape(-1)
    multipliers, shifts = [], []
    for ratio in real:
        significand, exponent = math.frexp(float(ratio))
        multiplier = round(significand * (1 << 31))
        if multiplier == 1 << 31:
            multiplier //= 2
            exponent += 1
        shift = 31 - exponent
        if shift < 0 or shift > INT32_MAX:
            raise ValueError(f'requant: 不支持 shift={shift}，要求非负 signed INT32。')
        multipliers.append(multiplier)
        shifts.append(shift)
    return checked_int32(multipliers, 'multiplier'), checked_int32(shifts, 'shift'), real


def integer_postprocess(accumulator, bias, multiplier, shift, leaky=True):
    """逐元素整数参考，最后一维为 Cout；Python 整数确保算术右移无溢出歧义。"""
    arrays = [np.asarray(value) for value in (accumulator, bias, multiplier, shift)]
    if any(not np.issubdtype(value.dtype, np.integer) for value in arrays):
        raise ValueError('PostProcess: 输入和参数必须是整数。')
    for value, name in zip(arrays, ('accumulator', 'bias', 'multiplier', 'shift')):
        checked_int32(value, name)
    if np.any(arrays[2] < 0) or np.any(arrays[3] < 0):
        raise ValueError('PostProcess: multiplier/shift 不得为负。')
    broadcast = np.broadcast_arrays(*arrays)
    output = np.empty(broadcast[0].shape, dtype=np.int8)
    for index in np.ndindex(output.shape):
        acc, b, m, s = (int(value[index]) for value in broadcast)
        a = acc + b
        if not INT32_MIN <= a <= INT32_MAX:
            raise OverflowError('PostProcess: accumulator+bias 超出 INT32。')
        if leaky and a < 0:
            m = (m * ALPHA_MULTIPLIER + (1 << 33)) >> ALPHA_SHIFT
        scaled = (a * m) >> s  # 不加 offset；负数算术右移向负无穷取整。
        output[index] = min(127, max(-128, scaled))
    return output


def convert_conv_weight_layout(weight):
    """OIHW → kh,kw,cin,cout → 逻辑 [Kh*Kw*Cin,Cout]，保持 INT8。"""
    if weight.ndim != 4 or weight.dtype != np.int8:
        raise ValueError('layout: 需要 OIHW INT8。')
    cout, cin, kh, kw = weight.shape
    return np.ascontiguousarray(weight.transpose(2, 3, 1, 0).reshape(kh * kw * cin, cout))


def pack_weight(logical, original_shape):
    """每个 kernel 位置单独补齐 Cin；Cout 按 4 分 Tile，每 Pack 为 4×8 字节。"""
    cout, cin, kh, kw = original_shape
    if logical.dtype != np.int8 or logical.shape != (kh * kw * cin, cout):
        raise ValueError('pack: 逻辑权重 shape/dtype 错误。')
    groups, tiles = (cin + 7) // 8, (cout + 3) // 4
    padded = np.zeros((kh * kw, groups * 8, tiles * 4), dtype=np.int8)
    padded[:, :cin, :cout] = logical.reshape(kh * kw, cin, cout)
    return np.ascontiguousarray(padded.reshape(kh * kw, groups, 8, tiles, 4)
                               .transpose(3, 0, 1, 4, 2).reshape(tiles, kh * kw * groups, 4, 8))


def validate_hardware_config(config, model):
    """第一版只接受固定 416/VOC20/5 anchors/125 通道及既定卷积结构。"""
    settings = config['model']
    if (settings['input_height'] != 416 or settings['input_width'] != 416
            or settings['num_classes'] != 20 or settings['num_anchors'] != 5
            or tuple(settings['classes']) != VOC_CLASSES
            or not np.array_equal(np.asarray(settings['anchors']), np.asarray(DEFAULT_VOC_ANCHORS))):
        raise ValueError('hardware: 仅支持 416 输入、VOC20 固定类别顺序、5 个既定 VOC anchors。')
    if not torch.equal(model.anchors.cpu(), torch.tensor(DEFAULT_VOC_ANCHORS, dtype=model.anchors.dtype)):
        raise ValueError('hardware: checkpoint anchors buffer 与固定 VOC 配置不符。')
    for index, name in enumerate(LAYER_NAMES):
        prediction = name == 'prediction'
        layer = model.prediction if prediction else getattr(model, name).conv
        kernel, padding = (1, 0) if prediction else (3, 1)
        expected = (CHANNELS[index+1], CHANNELS[index], kernel, kernel)
        if (tuple(layer.weight.shape) != expected or layer.stride != (1, 1)
                or layer.padding != (padding, padding) or layer.dilation != (1, 1) or layer.groups != 1):
            raise ValueError(f'{name}: 不支持的卷积 geometry，期望 OIHW={expected}。')
        if not prediction:
            block = getattr(model, name)
            if (not isinstance(block.bn, nn.Identity) or not isinstance(block.activation, nn.LeakyReLU)
                    or block.activation.negative_slope != 0.1):
                raise ValueError(f'{name}: 要求已折叠 BN 和 alpha=0.1 LeakyReLU。')


def collect_layers(model, config):
    """一次性验证所有层并准备数组；任何参数错误发生时还没有写出文件。"""
    validate_qat_structure(model)
    validate_hardware_config(config, model)
    input_scale, input_zero = extract_fake_quant_params(model.quant.activation_post_process, 'input')
    layers = []
    for name in LAYER_NAMES:
        try:
            prediction = name == 'prediction'
            block = getattr(model, name)
            layer = block if prediction else block.conv
            output_fq = layer.activation_post_process if prediction else block.activation.activation_post_process
            output_scale, output_zero = extract_fake_quant_params(output_fq, f'{name} output')
            weight_scale, weight_zero = extract_fake_quant_params(layer.weight_fake_quant, f'{name} weight', layer.out_channels)
            original = quantize_weight_per_channel(layer.weight, weight_scale, layer.weight_fake_quant)
            logical = convert_conv_weight_layout(original)
            packed = pack_weight(logical, original.shape)
            bias = convert_bias_to_int32(layer.bias, input_scale.item(), weight_scale)
            multiplier, shift, real = calculate_requant_params(input_scale, weight_scale, output_scale)
            layers.append(dict(name=name, original_shape=list(original.shape), leaky_relu=not prediction,
                               logical=logical, packed=packed, bias=bias, multiplier=multiplier, shift=shift,
                               weight_scale=weight_scale, weight_zero_point=weight_zero,
                               input_scale=input_scale.copy(), input_zero_point=input_zero.copy(),
                               output_scale=output_scale, output_zero_point=output_zero,
                               bias_scale=input_scale.item()*weight_scale, real_multiplier=real))
            input_scale, input_zero = output_scale, output_zero  # Pool 不改 scale。
        except (ValueError, OverflowError, RuntimeError) as error:
            raise ValueError(f'{name}: {error}') from error
    return layers


def write_layer(output, layer):
    """保存 NPY、原始 INT8 权重 BIN 和 little-endian field-first 参数 BIN。"""
    name = layer['name']
    arrays = {'weight': (f'weights/{name}_weight_int8.npy', layer['logical']),
              'bias': (f'biases/{name}_bias_int32.npy', layer['bias'])}
    for key in ('multiplier', 'shift'):
        arrays[key] = (f'quant_params/{name}_{key}_int32.npy', layer[key])
    for key in ('weight_scale', 'weight_zero_point', 'input_scale', 'input_zero_point',
                'output_scale', 'output_zero_point'):
        arrays[key] = (f'quant_params/{name}_{key}.npy', layer[key])
    files = {}
    for key, (relative, array) in arrays.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        files[key] = dict(path=relative, dtype=array.dtype.str, shape=list(array.shape),
                          order='C', byte_order='not_applicable' if array.itemsize == 1 else 'little',
                          data_nbytes=array.nbytes, file_size_bytes=path.stat().st_size, container='npy')
    params = np.concatenate([layer[key] for key in ('bias', 'multiplier', 'shift')]).astype('<i4')
    for key, relative, array in (('weight_packed', f'weights/{name}_weight_packed.bin', layer['packed']),
                                 ('params', f'params/{name}_params.bin', params)):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(array.tobytes(order='C'))
        if path.stat().st_size != array.nbytes:
            raise OSError(f'{name}: {relative} 文件字节数不符。')
        files[key] = dict(path=relative, dtype=array.dtype.str, shape=list(array.shape), order='C',
                          byte_order='not_applicable' if array.itemsize == 1 else 'little',
                          data_nbytes=array.nbytes, file_size_bytes=path.stat().st_size, container='raw')
    cout, cin, kh, kw = layer['original_shape']
    metadata = dict(name=name, original_oihw_shape=layer['original_shape'],
                    logical_weight_shape=list(layer['logical'].shape), packed_weight_shape=list(layer['packed'].shape),
                    Cin=cin, Cout=cout, G=(cin+7)//8, R=kh*kw*((cin+7)//8),
                    logical_input_channels=cin, physical_input_channels=((cin+7)//8)*8,
                    logical_output_channels=cout, physical_output_channels=((cout+3)//4)*4,
                    kernel=[kh, kw], stride=1, padding=0 if name == 'prediction' else 1,
                    leaky_relu=layer['leaky_relu'], weight_per_channel_axis=0,
                    logical_weight_order='kh,kw,cin,cout',
                    packed_weight_order='global_cout_tile,kh,kw,cin_group,cout_lane,cin_lane',
                    params_order=['bias', 'multiplier', 'shift'], params_count_per_field=cout,
                    params_field_byte_offsets=[0, cout*4, cout*8], files=files)
    for key in ('weight_scale', 'weight_zero_point', 'input_scale', 'input_zero_point', 'output_scale',
                'output_zero_point', 'bias_scale', 'multiplier', 'shift', 'real_multiplier'):
        metadata[key] = layer[key].tolist()
    return metadata


def export_checkpoint(checkpoint, output_dir):
    """严格恢复新 QAT，禁用 Observer，先校验所有层，再导出；不再次折叠已加载的 BN。"""
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    model = load_qat_model(checkpoint, torch.device('cpu')).eval()
    model.apply(disable_observer)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    layers = collect_layers(model, state['config'])
    metadata = dict(format_version=1, weight_layout_version=3, qat_scheme=QAT_SCHEME,
                    checkpoint=dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                    epoch=state.get('epoch')), model_config=state['config']['model'],
                    layer_order=list(LAYER_NAMES), pytorch_feature_layout='NCHW',
                    hardware_stream_order='H,W,channel_group8,channel_lane8', feature_ddr_layout='not_frozen',
                    datatypes=dict(input='int8', weight='int8', accumulator='int32', bias='int32', product='int64'),
                    quantization=dict(range=[-128, 127], activation='per_tensor_symmetric',
                                      weight='per_channel_symmetric', zero_point=0),
                    postprocess=dict(alpha_multiplier=ALPHA_MULTIPLIER, alpha_shift=ALPHA_SHIFT,
                                     negative_multiplier='(int64(M)*1717986918 + 2**33) >> 34',
                                     negative_multiplier_rounding='nearest_half_up',
                                     real_multiplier='input_scale * weight_scale / output_scale',
                                     multiplier_encoding='frexp_Q31; r ~= M / 2**S; S is total right shift',
                                     elementwise='clamp_int8((int64(accumulator+bias)*M_selected) >> S)',
                                     elementwise_rounding='arithmetic_right_shift_floor_no_offset',
                                     accumulator_plus_bias='must_fit_int32', offline_rounding='nearest_even'),
                    weight_byte_offset='32*(global_cout_tile*R + token) + 8*cout_lane + cin_lane',
                    weight_pack_bytes=32, weight_lane_word_endianness='little',
                    parameter_read_guard='check oc < Cout BEFORE reading bias/M/S')
    metadata['layers'] = [write_layer(output_dir, layer) for layer in layers]
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for layer in metadata['layers']:
        print(f"{layer['name']}: OIHW={layer['original_oihw_shape']} "
              f"logical={layer['logical_weight_shape']} packed={layer['packed_weight_shape']} "
              f"weight_bytes={layer['files']['weight_packed']['file_size_bytes']} "
              f"params_bytes={layer['files']['params']['file_size_bytes']}")
    print(f'导出完成：{output_dir}')
    return metadata


def main():
    """命令行默认读取本工程 qat_model.pth，支持独立 checkpoint/output 以便验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/qat_model.pth')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'generated')
    args = parser.parse_args()
    try:
        export_checkpoint(args.checkpoint, args.output_dir)
    except (ValueError, OverflowError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f'导出失败：{error}\n')


if __name__ == '__main__':
    main()
