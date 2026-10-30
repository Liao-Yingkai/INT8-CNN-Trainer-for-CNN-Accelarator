"""Class-aware NMS and VOC2007 eleven-point mAP at IoU 0.5."""
import argparse
from pathlib import Path
import torch
from torchvision.ops import batched_nms, box_iou
from torch.ao.quantization.fake_quantize import FakeQuantizeBase


@torch.no_grad()
def evaluate(model, loader, device, score_threshold=.01):
    """Accumulate detections and evaluate each class with difficult-object handling."""
    model.eval()
    observer_states = [(module, module.observer_enabled.clone()) for module in model.modules()
                       if isinstance(module, FakeQuantizeBase)]
    for module, _ in observer_states:
        module.disable_observer()
    records = {c: [] for c in range(model.num_classes)}
    ground_truth = {}
    image_id = 0
    for images, targets in loader:
        decoded = model.decode_predictions(model(images.to(device)))
        for b, target in enumerate(targets):
            boxes = decoded['boxes_xywh'][b].reshape(-1,4)
            boxes = torch.cat((boxes[:,:2]-boxes[:,2:]/2, boxes[:,:2]+boxes[:,2:]/2),1).clamp(0,1)
            scores = decoded['class_scores'][b].reshape(-1,model.num_classes)
            row, cls = torch.where(scores > score_threshold)
            chosen = batched_nms(boxes[row], scores[row,cls], cls, .45)[:300]
            for idx in chosen.tolist():
                c = int(cls[idx])
                records[c].append((float(scores[row[idx],c]), image_id, boxes[row[idx]].cpu()))
            ground_truth[image_id] = target
            image_id += 1
    aps = []
    for c in range(model.num_classes):
        total = sum(int(((t['boxes'][:,4] == c) & ~t['difficult']).sum()) for t in ground_truth.values())
        matched, tp, fp = set(), [], []
        for _, identifier, box in sorted(records[c], key=lambda x: -x[0]):
            target = ground_truth[identifier]
            mask = target['boxes'][:,4] == c
            gt, difficult = target['boxes'][mask,:4], target['difficult'][mask]
            if len(gt):
                iou, index = box_iou(box[None], gt)[0].max(0)
                if iou >= .5:
                    index = int(index)
                    if difficult[index]:
                        continue
                    key = (identifier,index)
                    tp.append(float(key not in matched)); fp.append(float(key in matched))
                    matched.add(key)
                    continue
            tp.append(0.); fp.append(1.)
        tp, fp = torch.tensor(tp).cumsum(0), torch.tensor(fp).cumsum(0)
        recall, precision = tp / max(total,1), tp / (tp+fp).clamp_min(1)
        ap = sum(float(precision[recall >= threshold].max()) if (recall >= threshold).any() else 0.
                 for threshold in torch.linspace(0,1,11)) / 11
        aps.append(ap)
    for module, enabled in observer_states:
        module.observer_enabled.copy_(enabled)
    return sum(aps)/len(aps), aps


def main():
    """Evaluate a saved floating-point checkpoint on configured evaluation split."""
    from train import load_config, make_loader, build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['float', 'qat', 'both'], default='both')
    args = parser.parse_args()
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from quant import load_qat_model
    scores = {}
    modes = ['float', 'qat'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        path = Path(__file__).parent / f'checkpoints/{mode}_model.pth'
        state = torch.load(path, map_location='cpu', weights_only=False)
        config['model'] = state['config']['model']
        if mode == 'qat':
            model = load_qat_model(path, device)
        else:
            model = build_model(config).to(device)
            model.load_state_dict(state['model_state_dict'])
        score, aps = evaluate(model, make_loader(config, 'evaluation', False), device)
        scores[mode] = score
        for name, ap in zip(config['model']['classes'], aps):
            print(f'{mode} {name}: AP50={ap:.4f}')
        print(f'{mode} VOC2007 mAP50={score:.4f}')
        del model
    if len(scores) == 2:
        print(f'QAT - FP32 mAP50={scores["qat"]-scores["float"]:+.4f}')


if __name__ == '__main__':
    main()
