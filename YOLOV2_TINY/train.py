"""Floating-point VOC training entry point."""
import argparse
from pathlib import Path
import random
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import VOCDataset, collate
from model import YOLOv2Tiny
from loss import YOLOLoss
from evaluate import evaluate

ROOT = Path(__file__).resolve().parent


def load_config():
    """Read the project configuration."""
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    model = config['model']
    if model['num_classes'] != len(model['classes']):
        raise ValueError('num_classes must match the class-name list.')
    if model['num_anchors'] != len(model['anchors']):
        raise ValueError('num_anchors must match the anchor list.')
    return config


def build_model(config):
    """Construct the detector using the configured anchors."""
    settings = config['model']
    return YOLOv2Tiny(settings['num_classes'], settings['num_anchors'], settings['anchors'])


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


def main():
    """Train FP32 weights, validate mAP and save resumable last/best checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['float'], default='float')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = load_config()
    settings = config['training']
    random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    model = build_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'],
                                momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings['milestones'], gamma=.1)
    start, best = 0, -1.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start, best = checkpoint['epoch'], checkpoint['best_map']
    training = make_loader(config, 'train', True)
    validation = make_loader(config, 'validation', False)
    criterion = YOLOLoss()
    output = ROOT / 'checkpoints'
    output.mkdir(exist_ok=True)
    for epoch in range(start, settings['epochs']):
        model.train()
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
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), epoch=epoch+1, best_map=best,
                     map50=score, config=config)
        torch.save(state, output / 'float_model.pth')
        if improved:
            torch.save(state, output / 'float_model_best.pth')
        print(f'epoch={epoch+1}/{settings["epochs"]} loss={total/max(count,1):.5f} mAP50={score:.4f}')


if __name__ == '__main__':
    main()
