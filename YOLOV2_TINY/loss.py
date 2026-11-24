"""YOLOv2-style anchor assignment and detection loss."""
import torch
from torch import nn
from torchvision.ops import box_iou


class YOLOLoss(nn.Module):
    """Use coordinate regression, IoU confidence targets and class cross entropy."""
    def forward(self, logits, targets, model):
        """Assign each object to its best shape anchor; ignore overlapping negatives."""
        raw = model.reshape_predictions(logits)
        decoded = model.decode_predictions(logits)
        boxes = decoded['boxes_xywh'].detach()
        xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                          boxes[..., :2] + boxes[..., 2:] / 2), -1)
        n, a, h, w, _ = raw.shape
        positive = torch.zeros(raw.shape[:-1], dtype=torch.bool, device=raw.device)
        negative = torch.ones_like(positive)
        truth = torch.zeros_like(raw[..., :4])
        labels = torch.zeros_like(positive, dtype=torch.long)
        confidence = torch.zeros_like(raw[..., 4])
        for b, target in enumerate(targets):
            gt = target['boxes'].to(raw.device)
            gt = gt[~target['difficult'].to(raw.device)]
            if not len(gt):
                continue
            overlaps = box_iou(xyxy[b].reshape(-1, 4), gt[:, :4])
            negative[b] = (overlaps.max(1).values < .6).reshape(a, h, w)
            wh = (gt[:, 2:4] - gt[:, :2]) * raw.new_tensor([w, h])
            center = (gt[:, :2] + gt[:, 2:4]) / 2 * raw.new_tensor([w, h])
            anchors = model.anchors.to(raw.device)
            inter = torch.minimum(wh[:, None], anchors[None]).prod(-1)
            best = (inter / (wh.prod(-1)[:, None] + anchors.prod(-1)[None] - inter)).argmax(1)
            for j in range(len(gt)):
                x, y = center[j].long().tolist()
                x, y = min(x, w-1), min(y, h-1)
                k = int(best[j])
                positive[b,k,y,x] = True
                truth[b,k,y,x,:2] = center[j] - raw.new_tensor([x,y])
                truth[b,k,y,x,2:] = torch.log(wh[j] / anchors[k])
                labels[b,k,y,x] = gt[j,4].long()
                confidence[b,k,y,x] = overlaps.reshape(a,h,w,-1)[k,y,x,j]
        zero = raw.sum() * 0
        coord = zero
        classification = zero
        if positive.any():
            coord = ((raw[..., :2].sigmoid()[positive] - truth[..., :2][positive]) ** 2).sum()
            coord += ((raw[..., 2:4][positive] - truth[..., 2:4][positive]) ** 2).sum()
            classification = nn.functional.cross_entropy(raw[...,5:][positive], labels[positive], reduction='sum')
        error = (raw[...,4].sigmoid() - confidence) ** 2
        obj = 5 * error[positive].sum() + error[negative & ~positive].sum()
        return (coord + obj + classification) / n
