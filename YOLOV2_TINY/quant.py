"""Official eager QAT preparation and portable checkpoint helpers."""
from pathlib import Path
import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize, MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver, QConfig, prepare_qat,
)
from model import YOLOv2Tiny, DarknetMaxPool2d


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
    """Freeze/fold pretrained BN and prepare official fake quantization modules."""
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = create_fpga_qconfig()  # type: ignore
    # LeakyReLU is not part of the standard Conv+BN+ReLU fuser. Keep it as
    # a separate quantization boundary, and leave padding/pooling unobserved.
    for module in model.modules():
        if isinstance(module, (nn.Identity, DarknetMaxPool2d)):
            module.qconfig = None  # type: ignore
    prepare_qat(model, inplace=True)
    return model


def prepare_qat_model(float_checkpoint, device):
    """Initialize QAT from FP32 weights and the checkpoint's model configuration."""
    state = torch.load(float_checkpoint, map_location='cpu', weights_only=False)
    model = build_model(state['config'])
    model.load_state_dict(state['model_state_dict'])
    return _prepare_structure(model).to(device)


def load_qat_model(qat_checkpoint, device):
    """Restore fused QAT structure before loading fake-quant and weight state."""
    state = torch.load(qat_checkpoint, map_location='cpu', weights_only=False)
    model = _prepare_structure(build_model(state['config']))
    model.load_state_dict(state['model_state_dict'])
    return model.to(device)


def save_checkpoint(path, model, epoch, map50, config):
    """Save model state, detection metric and construction configuration."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                    map50=map50, config=config), path)
