"""VOC detection data with letterbox preprocessing."""
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class VOCDataset(Dataset):
    """Read local VOCdevkit directories and return normalized xyxy boxes."""
    def __init__(self, root, splits, classes, size=(416, 416), augment=False):
        """Collect image identifiers from year/split pairs."""
        self.samples = []
        self.classes, self.size, self.augment = classes, size, augment
        for year, split in splits:
            folder = Path(root) / f"VOC{year}"
            ids = (folder / 'ImageSets/Main' / f'{split}.txt').read_text().split()
            self.samples.extend((folder, item) for item in ids)

    def __len__(self):
        """Return image count."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load RGB image and transform boxes with the same resize and padding."""
        folder, identifier = self.samples[index]
        image = Image.open(folder / 'JPEGImages' / f'{identifier}.jpg').convert('RGB')
        rows, difficult = [], []
        for obj in ET.parse(folder / 'Annotations' / f'{identifier}.xml').findall('object'):
            name = obj.findtext('name')
            if name not in self.classes:
                continue
            box = obj.find('bndbox')
            rows.append([float(box.findtext(k)) - (1 if k in ('xmin', 'ymin') else 0)
                         for k in ('xmin', 'ymin', 'xmax', 'ymax')] + [self.classes.index(name)])
            difficult.append(int(obj.findtext('difficult', '0')))
        targets = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
        width, height = image.size
        if self.augment:
            image = ImageEnhance.Color(image).enhance(random.uniform(.7, 1.3))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.7, 1.3))
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                targets[:, [0, 2]] = width - targets[:, [2, 0]]
        out_h, out_w = self.size
        ratio = min(out_w / width, out_h / height)
        rw, rh = round(width * ratio), round(height * ratio)
        left, top = (out_w - rw) // 2, (out_h - rh) // 2
        canvas = Image.new('RGB', (out_w, out_h), (128, 128, 128))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
        targets[:, [0, 2]] = (targets[:, [0, 2]] * rw / width + left) / out_w
        targets[:, [1, 3]] = (targets[:, [1, 3]] * rh / height + top) / out_h
        targets[:, :4].clamp_(0, 1)
        valid = (targets[:, 2] > targets[:, 0]) & (targets[:, 3] > targets[:, 1])
        return to_tensor(canvas), {'boxes': targets[valid], 'difficult': torch.tensor(difficult, dtype=torch.bool)[valid]}


def collate(batch):
    """Stack images while preserving variable object counts."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
