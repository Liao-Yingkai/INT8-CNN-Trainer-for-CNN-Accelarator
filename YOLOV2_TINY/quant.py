"""Official eager QAT preparation and portable checkpoint helpers."""
from pathlib import Path
import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize, MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver, QConfig, prepare_qat,
)
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from model import YOLOv2Tiny

QAT_SCHEME = 'single_postprocess_v1'


def create_fpga_qconfig():
    """Use symmetric INT8 activations and output-channel INT8 weights."""
    return QConfig(
        activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False),
        weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric, ch_axis=0),
    )


def build_model(config):
    """Construct the same network from checkpoint or YAML model settings."""
    settings = config['model']
    return YOLOv2Tiny(settings['num_classes'], settings['num_anchors'], settings['anchors'])


def _prepare_structure(model):
    """先折叠 BN；只对输入、LeakyReLU 后和 prediction 输出做激活量化。"""
    model.eval()
    model.fuse_model()
    model.train()
    config = create_fpga_qconfig()
    # 与本机官方 default_weight_only_qconfig 相同，Identity 不观察或量化激活；
    # 不能把 Conv.qconfig 设为 None，否则 prepare_qat 不会创建 weight FakeQuant。
    weight_only = QConfig(activation=nn.Identity, weight=config.weight)
    # 默认关闭继承，仅显式启用白名单叶子，容器、BN Identity 和 Pool 均不量化。
    for module in model.modules():
        module.qconfig = None  # type: ignore
    model.quant.qconfig = config  # type: ignore
    for index in range(1, 9):
        block = getattr(model, f'conv{index}')
        block.conv.qconfig = weight_only  # type: ignore
        block.activation.qconfig = config  # type: ignore
    model.prediction.qconfig = config  # type: ignore
    prepare_qat(model, inplace=True)
    model.qat_scheme = QAT_SCHEME
    validate_qat_structure(model)
    return model


def validate_qat_structure(model):
    """核实真实模块而非配置名称：9 个权重 FQ、10 个激活 FQ，无额外边界。"""
    expected = {'quant.activation_post_process', 'prediction.activation_post_process',
                'prediction.weight_fake_quant'}
    for index in range(1, 9):
        block = getattr(model, f'conv{index}')
        if not isinstance(block.conv.activation_post_process, nn.Identity):
            raise ValueError(f'conv{index}: 激活前必须是 Identity，不能保留 activation FakeQuant。')
        expected.update({f'conv{index}.conv.weight_fake_quant',
                         f'conv{index}.activation.activation_post_process'})
    actual = {name for name, module in model.named_modules()
              if isinstance(module, FakeQuantizeBase)}
    if actual != expected or getattr(model, 'qat_scheme', None) != QAT_SCHEME:
        raise ValueError(f'QAT 结构/方案错误；缺失 {expected-actual}，多余 {actual-expected}。')


def prepare_qat_model(float_checkpoint, device):
    """Initialize QAT from FP32 weights and the checkpoint's model configuration."""
    state = torch.load(float_checkpoint, map_location='cpu', weights_only=False)
    if state.get('qat_scheme') is not None or any('weight_fake_quant' in key
                                                for key in state['model_state_dict']):
        raise ValueError('QAT 初始化必须使用未融合的 FP32 checkpoint，而不是 QAT checkpoint。')
    model = build_model(state['config'])
    model.load_state_dict(state['model_state_dict'])
    return _prepare_structure(model).to(device)


def load_qat_model(qat_checkpoint, device):
    """Restore fused QAT structure before loading fake-quant and weight state."""
    state = torch.load(qat_checkpoint, map_location='cpu', weights_only=False)
    if state.get('qat_scheme') != QAT_SCHEME:
        raise ValueError(f'不兼容的 QAT checkpoint：要求 qat_scheme={QAT_SCHEME}；'
                         '旧双边界 QAT / FP32 不可直接加载，请从 FP32 重新进行 QAT 微调。')
    model = _prepare_structure(build_model(state['config']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device)


def save_checkpoint(path, model, epoch, map50, config):
    """保存模型及评估信息；仅经结构检查的 QAT 才写入新方案标记。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = dict(model_state_dict=model.state_dict(), epoch=epoch, map50=map50, config=config)
    if any(isinstance(module, FakeQuantizeBase) for module in model.modules()):
        validate_qat_structure(model)
        state['qat_scheme'] = QAT_SCHEME
    torch.save(state, path)
