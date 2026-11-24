"""Floating-point VOC training entry point."""
import argparse
from pathlib import Path
import random
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import VOCDataset, collate
from loss import YOLOLoss
from evaluate import evaluate
from quant import build_model, prepare_qat_model, save_checkpoint
from torch.ao.quantization import disable_observer

ROOT = Path(__file__).resolve().parent


def load_config(config_path=None):
    """Read the project configuration."""
    config = yaml.safe_load(Path(config_path or ROOT / 'config.yaml').read_text(encoding='utf-8'))
    model = config['model']
    if model['num_classes'] != len(model['classes']):
        raise ValueError('num_classes must match the class-name list.')
    if model['num_anchors'] != len(model['anchors']):
        raise ValueError('num_anchors must match the anchor list.')
    return config


def select_device(name='auto'):
    """自动选择设备或尊重显式配置，不依赖特定计算设备。"""
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is not available; select --device cpu.')
    return torch.device(name)


def make_loader(config, split, augment):
    """Build a VOC loader with variable-length detection targets."""
    settings = config['model']
    data_root = Path(config['data']['root'])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    dataset = VOCDataset(data_root, config['data'][split], settings['classes'],
                         (settings['input_height'], settings['input_width']), augment)
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=augment,
                      num_workers=config['training']['num_workers'], collate_fn=collate,
                      pin_memory=torch.cuda.is_available())


def train_model(model, training, validation, device, config, mode):
    """Train FP32 or fine-tune QAT and save final/best weights like MNIST."""
    settings = config['training']
    epochs = settings['float_epochs'] if mode == 'float' else settings['qat_epochs']
    lr = settings['float_learning_rate'] if mode == 'float' else settings['qat_learning_rate']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=.9, weight_decay=.0005)
    milestones = settings['float_milestones'] if mode == 'float' else settings['qat_milestones']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=.1)
    best = -1.
    criterion = YOLOLoss()
    output = ROOT / 'checkpoints'
    output.mkdir(exist_ok=True)
    for epoch in range(epochs):
        model.train()
        if mode == 'qat' and epoch >= settings['qat_observer_freeze_epoch']:
            model.apply(disable_observer)
        total, count = 0., 0
        for images, targets in training:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), targets, model)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch+1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += loss.item() * len(images); count += len(images)
        scheduler.step()
        score, _ = evaluate(model, validation, device)
        improved = score > best
        best = max(best, score)
        save_checkpoint(output / f'{mode}_model.pth', model, epoch+1, score, config)
        if improved:
            save_checkpoint(output / f'{mode}_model_best.pth', model, epoch+1, score, config)
        print(f'epoch={epoch+1}/{epochs} loss={total/max(count,1):.5f} mAP50={score:.4f}')
    return best


def main():
    """Run FP32 training or lower-learning-rate QAT initialized from FP32."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['float', 'qat'], required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config.yaml')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--float-checkpoint', type=Path, default=ROOT/'checkpoints/float_model.pth')
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config['training']['seed']); torch.manual_seed(config['training']['seed'])
    device = select_device(args.device)
    print(f'Using {device}')
    float_path = args.float_checkpoint
    if args.mode == 'qat':
        state = torch.load(float_path, map_location='cpu', weights_only=False)
        config['model'] = state['config']['model']
        model = build_model(config).to(device)
        model.load_state_dict(state['model_state_dict'])
    else:
        model = build_model(config).to(device)
    training = make_loader(config, 'train', True)
    validation = make_loader(config, 'validation', False)
    if args.mode == 'float':
        train_model(model, training, validation, device, config, 'float')
        return
    fp32_map, _ = evaluate(model, validation, device)
    del model
    qat = prepare_qat_model(float_path, device)
    qat_map = train_model(qat, training, validation, device, config, 'qat')
    print(f'FP32 mAP50={fp32_map:.4f} | best QAT mAP50={qat_map:.4f} | delta={qat_map-fp32_map:+.4f}')


if __name__ == '__main__':
    main()
