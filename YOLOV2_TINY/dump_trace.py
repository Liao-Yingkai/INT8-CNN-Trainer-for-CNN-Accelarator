"""读取导出的权重/参数 BIN，生成 VOC 测试图的模块边界整数参考 BIN。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset import VOCDataset
from export import (ALPHA_MULTIPLIER, ALPHA_SHIFT, CHANNELS, LAYER_NAMES, VOC_CLASSES,
                    checked_int32, integer_postprocess, positive_scale)
from model import DEFAULT_VOC_ANCHORS
from quant import QAT_SCHEME

ROOT = Path(__file__).resolve().parent


def read_binary(root, descriptor, dtype, shape):
    """核对原始 BIN 的 dtype、shape 和字节数，避免将旧格式参数误当作新文件。"""
    dtype = np.dtype(dtype)
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    path = root/descriptor['path']
    if (descriptor['container'] != 'raw' or np.dtype(descriptor['dtype']) != dtype
            or descriptor['shape'] != list(shape) or descriptor['order'] != 'C'
            or descriptor['data_nbytes'] != expected_bytes
            or descriptor['file_size_bytes'] != expected_bytes
            or path.stat().st_size != expected_bytes):
        raise ValueError(f'{path}: BIN 描述或实际长度不符。')
    if dtype.itemsize > 1 and descriptor['byte_order'] != 'little':
        raise ValueError(f'{path}: INT32 BIN 必须是 little-endian。')
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_fixed_layers(generated_dir):
    """只读 metadata v2 和三类 BIN；不读取 NPY、不恢复 checkpoint、不更新 scale。"""
    root = Path(generated_dir)
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('format_version') != 2 or metadata.get('weight_layout_version') != 3
            or metadata.get('qat_scheme') != QAT_SCHEME or metadata.get('layer_order') != list(LAYER_NAMES)):
        raise ValueError('要求 metadata v2、weight layout v3、single_postprocess_v1，请重新导出参数。')
    config = metadata['model_config']
    quantization = metadata['quantization']
    if (quantization['range'] != [-128, 127] or quantization['zero_point'] != 0
            or quantization['activation'] != 'per_tensor_symmetric'
            or quantization['weight'] != 'per_channel_symmetric'):
        raise ValueError('仅支持 signed symmetric INT8，zero_point=0。')
    if (config['input_height'] != 416 or config['input_width'] != 416
            or config['num_classes'] != 20 or config['num_anchors'] != 5
            or tuple(config['classes']) != VOC_CLASSES
            or not np.array_equal(config['anchors'], DEFAULT_VOC_ANCHORS)):
        raise ValueError('Trace 仅支持固定 416/VOC20/5 anchors 网络。')
    contract = metadata['postprocess']
    if (contract['alpha_multiplier'] != ALPHA_MULTIPLIER or contract['alpha_shift'] != ALPHA_SHIFT
            or contract['elementwise_rounding'] != 'arithmetic_right_shift_floor_no_offset'):
        raise ValueError('metadata 的 alpha 或右移契约与整数参考不符。')
    entries = metadata['layers']
    if [entry['name'] for entry in entries] != list(LAYER_NAMES):
        raise ValueError('metadata 层顺序或层数错误。')
    layers, previous_scale = [], None
    for index, entry in enumerate(entries):
        name = entry['name']
        try:
            cin, cout = CHANNELS[index:index+2]
            kernel, padding = (1, 0) if name == 'prediction' else (3, 1)
            groups, tiles = (cin+7)//8, (cout+3)//4
            shape = (tiles, kernel*kernel*groups, 4, 8)
            if (entry['original_oihw_shape'] != [cout, cin, kernel, kernel]
                    or entry['packed_weight_shape'] != list(shape)
                    or entry['logical_weight_shape'] != [kernel*kernel*cin, cout]
                    or entry['kernel'] != [kernel, kernel] or entry['padding'] != padding
                    or entry['stride'] != 1 or entry['leaky_relu'] != (name != 'prediction')
                    or entry['Cin'] != cin or entry['Cout'] != cout
                    or entry['G'] != groups or entry['R'] != shape[1]
                    or entry['weight_per_channel_axis'] != 0):
                raise ValueError('固定卷积 geometry 或物理 shape 不符。')
            for key, count in (('input', 1), ('output', 1), ('weight', cout)):
                if np.asarray(entry[f'{key}_zero_point']).shape != (count,) or np.any(np.asarray(entry[f'{key}_zero_point']) != 0):
                    raise ValueError(f'{key}: 要求对应长度的 zero_point=0。')
            ins = positive_scale(entry['input_scale'], f'{name} input')
            outs = positive_scale(entry['output_scale'], f'{name} output')
            if ins.shape != (1,) or outs.shape != (1,) or (previous_scale is not None and ins.item() != previous_scale):
                raise ValueError('activation scale 必须是标量，且层间/Pool 的 scale 必须连续。')
            files = entry['files']
            packed = read_binary(root, files['weight_bin'], 'i1', shape)
            padded = packed.reshape(tiles, kernel*kernel, groups, 4, 8).transpose(1, 2, 4, 0, 3)
            padded = padded.reshape(kernel*kernel, groups*8, tiles*4)
            if np.any(padded[:, cin:, :] != 0) or np.any(padded[:, :, cout:] != 0):
                raise ValueError('权重 Cin/Cout 尾部不是 0。')
            weight = np.ascontiguousarray(padded[:, :cin, :cout].reshape(kernel*kernel*cin, cout))
            bias = read_binary(root, files['bias_bin'], '<i4', (cout,))
            quant = read_binary(root, files['quant_bin'], '<i4', (2*cout,)).reshape(2, cout)
            if (files['bias_bin']['fields'] != ['bias'] or files['bias_bin']['field_byte_offsets'] != [0]
                    or files['quant_bin']['fields'] != ['multiplier', 'shift']
                    or files['quant_bin']['field_byte_offsets'] != [0, 4*cout]
                    or files['bias_bin']['count_per_field'] != cout
                    or files['quant_bin']['count_per_field'] != cout
                    or np.any(quant < 0)
                    or not np.array_equal(quant[0], entry['multiplier'])
                    or not np.array_equal(quant[1], entry['shift'])):
                raise ValueError('bias/quant 的 field-first 布局、数量或非负 M/S 错误。')
            layers.append(dict(name=name, kernel=kernel, padding=padding, weight=weight,
                               bias=bias, multiplier=quant[0], shift=quant[1],
                               leaky=name != 'prediction', input_scale=ins.item()))
            previous_scale = outs.item()
        except (ValueError, KeyError, OSError) as error:
            raise ValueError(f'{name}: {error}') from error
    return layers


def quantize_input(image_hwc, scale):
    """RGB [0,1] 输入按训练 FQ 的 FP32 reciprocal、最近偶数和饱和规则量化。"""
    positive_scale(scale, 'input')
    values = torch.as_tensor(np.ascontiguousarray(image_hwc), dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError('输入图像包含非有限值。')
    inverse = torch.tensor(scale, dtype=torch.float32).reciprocal()
    return torch.round(values*inverse).clamp(-128, 127).to(torch.int8).numpy()


def unfold_hwc(feature, kernel, padding):
    """标准 unfold 后重排为 [OH*OW,kh*kw*Cin]，INT8 和空间零 Padding 均精确保留。"""
    if feature.dtype != np.int8 or feature.ndim != 3 or (kernel, padding) not in ((3, 1), (1, 0)):
        raise ValueError('unfold: 要求 HWC INT8，以及 3x3/pad1 或 1x1/pad0。')
    height, width, channels = feature.shape
    tensor = torch.from_numpy(np.ascontiguousarray(feature.transpose(2, 0, 1))).float().unsqueeze(0)
    patches = F.unfold(tensor, kernel_size=kernel, padding=padding)
    rows = patches.reshape(channels, kernel, kernel, height*width).permute(3, 1, 2, 0)
    return rows.reshape(height*width, kernel*kernel*channels).to(torch.int8).contiguous().numpy()


def pack_im2row(rows, height, width, channels, kernel):
    """[H,T,R,4,8]：按行切空间 Tile，每个 kernel 位置独立补齐 Cin，所有无效 Lane 为 0。"""
    if rows.dtype != np.int8 or rows.shape != (height*width, kernel*kernel*channels):
        raise ValueError('im2row: unfold shape/dtype 错误。')
    groups, tiles = (channels+7)//8, (width+3)//4
    padded = np.zeros((height, tiles*4, kernel*kernel, groups*8), dtype=np.int8)
    padded[:, :width, :, :channels] = rows.reshape(height, width, kernel*kernel, channels)
    return np.ascontiguousarray(padded.reshape(height, tiles, 4, kernel*kernel, groups, 8)
                               .transpose(0, 1, 3, 4, 2, 5).reshape(height, tiles, kernel*kernel*groups, 4, 8))


def integer_mac(rows, weight):
    """整数操作数用 CPU float64 GEMM 精确求和，分行块计算并检查 INT32，不模拟阵列时序。"""
    if rows.dtype != np.int8 or weight.dtype != np.int8 or rows.ndim != 2 or weight.ndim != 2 or rows.shape[1] != weight.shape[0]:
        raise ValueError('MAC: 需要兼容的两个 INT8 矩阵。')
    # 所有乘积及任意部分和的绝对界 K*128*128 < 2**53，因此 double 的计算严格为整数。
    # 固定 YOLO 最大 K=9216，最坏界也小于 INT32 上界；保留显式范围检查。
    if rows.shape[1]*128*128 >= 2**53:
        raise ValueError('MAC: 超出 float64 的整数精确表示界。')
    matrix = torch.from_numpy(weight).double()
    output = np.empty((rows.shape[0], weight.shape[1]), dtype=np.int32)
    for start in range(0, len(rows), 4096):
        product = torch.from_numpy(rows[start:start+4096]).double() @ matrix
        if not torch.equal(product, product.round()):
            raise ValueError('MAC: 检测到非整数结果。')
        output[start:start+4096] = checked_int32(product.numpy(), 'systolic accumulator')
    return output


def max_pool_hwc(feature, stride):
    """INT8 2x2 MaxPool；stride1 只在右/下补 -128，stride2 不补边。"""
    if feature.dtype != np.int8 or feature.ndim != 3 or stride not in (1, 2):
        raise ValueError('Pool: 要求 HWC INT8 和 stride=1/2。')
    if stride == 1:
        feature = np.pad(feature, ((0, 1), (0, 1), (0, 0)), constant_values=-128)
    windows = np.lib.stride_tricks.sliding_window_view(feature, (2, 2), axis=(0, 1))
    return np.ascontiguousarray(windows[::stride, ::stride].max(axis=(-2, -1)))


def stage_names():
    """固定 43 个真实模块边界；bias/LeakyReLU/requant 不再分别导出。"""
    names = ['input_int8_hwc']
    for index, name in enumerate(LAYER_NAMES):
        names.extend(f'{name}_{suffix}' for suffix in ('unfold_int8', 'im2row_int8',
                     'systolic_int32_hwc', 'postprocess_int8_hwc'))
        if index < 6:
            names.append(f'pool{index+1}_int8_hwc')
    return names


def run_fixed_model(image_hwc, layers):
    """逐层产出参考数组以控制内存；下一层只消费整数参考结果，不混入 QAT 输出。"""
    feature = quantize_input(image_hwc, layers[0]['input_scale'])
    yield 'input_int8_hwc', feature
    for index, layer in enumerate(layers):
        name, kernel = layer['name'], layer['kernel']
        height, width, channels = feature.shape
        rows = unfold_hwc(feature, kernel, layer['padding'])
        yield f'{name}_unfold_int8', rows
        yield f'{name}_im2row_int8', pack_im2row(rows, height, width, channels, kernel)
        accumulator = integer_mac(rows, layer['weight']).reshape(height, width, -1)
        yield f'{name}_systolic_int32_hwc', accumulator
        # 此函数同时做 bias、固定 alpha 负分支、算术右移和饱和。中途不落 INT8。
        feature = integer_postprocess(accumulator, layer['bias'], layer['multiplier'],
                                      layer['shift'], layer['leaky'])
        yield f'{name}_postprocess_int8_hwc', feature
        if index < 6:
            feature = max_pool_hwc(feature, 2 if index < 5 else 1)
            yield f'pool{index+1}_int8_hwc', feature


def select_test_indices(dataset_size, num_samples, seed):
    """与 MNIST 一样按固定 seed 无放回抽样，序号后缀表示导出顺序而非数据集 ID。"""
    if num_samples < 1 or num_samples > dataset_size:
        raise ValueError(f'num_samples 必须在 1..{dataset_size} 内，不能重复抽样。')
    return [int(index) for index in np.random.default_rng(seed).choice(dataset_size, num_samples, replace=False)]


def trace_path(output_dir, name, sequence):
    """所有 trace BIN 平铺存放，采用 trace_<边界>_<四位序号>.bin。"""
    return Path(output_dir)/f'trace_{name}_{sequence:04d}.bin'


def check_output_paths(output_dir, num_samples):
    """所有待写路径提前检查；不清空旧 trace，不覆盖已有文件或参数目录。"""
    for sequence in range(num_samples):
        for name in stage_names():
            path = trace_path(output_dir, name, sequence)
            if path.exists():
                raise FileExistsError(f'{path} 已存在，请使用新的 --output-dir；不会覆盖或删除旧 trace。')


def write_trace_binary(path, name, array):
    """保存补码 INT8 或明确 little-endian INT32，无文件头、无结构体、无新 metadata。"""
    expected = np.dtype(np.int32 if 'int32' in name else np.int8)
    if array.dtype != expected:
        raise TypeError(f'{name}: 预期 {expected}，实际 {array.dtype}。')
    array = np.ascontiguousarray(array, dtype='<i4' if expected.itemsize == 4 else 'i1')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(array.tobytes(order='C'))
    if path.stat().st_size != array.nbytes:
        raise OSError(f'{path}: 文件字节数不正确。')


def main():
    """从本地 VOC2007 test 抽样，生成全部 9 层的整数模块参考，不训练、不下载。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--num-samples', '--num-images', dest='num_samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--generated-dir', type=Path, default=ROOT/'generated')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        output = args.output_dir if args.output_dir is not None else args.generated_dir/'trace'
        layers = load_fixed_layers(args.generated_dir)
        config = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
        data_root = Path(config['data']['root'])
        if not data_root.is_absolute():
            data_root = ROOT/data_root
        dataset = VOCDataset(data_root, [(2007, 'test')], list(VOC_CLASSES), (416, 416), augment=False)
        indices = select_test_indices(len(dataset), args.num_samples, args.seed)
        check_output_paths(output, args.num_samples)
        print(f'参数目录={args.generated_dir.resolve()}, trace目录={Path(output).resolve()}', flush=True)
        print(f'VOC2007 test seed={args.seed}, indices={indices}', flush=True)
        for sequence, sample_index in enumerate(indices):
            image, _ = dataset[sample_index]
            identifier = dataset.samples[sample_index][1]
            print(f'[{sequence+1}/{len(indices)}] sequence={sequence:04d}, '
                  f'test index={sample_index}, image={identifier}', flush=True)
            for name, array in run_fixed_model(image.permute(1, 2, 0).contiguous().numpy(), layers):
                path = trace_path(output, name, sequence)
                write_trace_binary(path, name, array)
                print(f'  {path.name}: shape={list(array.shape)}, dtype={array.dtype}, bytes={array.nbytes}', flush=True)
        print(f'Trace 导出完成：{output}', flush=True)
    except (ValueError, TypeError, OverflowError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Trace 导出失败：{error}\n')


if __name__ == '__main__':
    main()
