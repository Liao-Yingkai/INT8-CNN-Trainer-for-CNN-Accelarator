# YOLOv2-tiny VOC PyTorch 工程

当前已提供浮点网络、VOC 数据加载、检测损失、训练和 VOC2007 mAP50 评估。QAT 和 FPGA 参数导出在后续阶段接入。

## 网络规格

`model.py` 实现 Darknet `yolov2-tiny-voc.cfg` 的主干和原始卷积预测头：

| 阶段 | 操作 | 输出通道 / 网格（输入为 416×416 时） |
| --- | --- | --- |
| Input | RGB | `3 × 416 × 416` |
| Conv1 + Pool1 | `3×3, 16`；`2×2 / 2` | `16 × 208 × 208` |
| Conv2 + Pool2 | `3×3, 32`；`2×2 / 2` | `32 × 104 × 104` |
| Conv3 + Pool3 | `3×3, 64`；`2×2 / 2` | `64 × 52 × 52` |
| Conv4 + Pool4 | `3×3, 128`；`2×2 / 2` | `128 × 26 × 26` |
| Conv5 + Pool5 | `3×3, 256`；`2×2 / 2` | `256 × 13 × 13` |
| Conv6 + Pool6 | `3×3, 512`；`2×2 / 1` | `512 × 13 × 13` |
| Conv7 | `3×3, 1024` | `1024 × 13 × 13` |
| Conv8 | `3×3, 1024` | `1024 × 13 × 13` |
| Prediction | `1×1, 125` | `125 × 13 × 13` |

前 8 个卷积均为 Conv + BatchNorm + LeakyReLU（斜率 `0.1`）；prediction 卷积不带 BatchNorm 和激活函数，返回 raw logits。VOC 检测头默认 `5 × (5 + 20) = 125` 个通道。模型输入为 PyTorch 标准 NCHW；输入高宽须是 32 的倍数。`reshape_predictions()` 可将输出整理为 `[N, anchors, grid_h, grid_w, 5 + classes]`，供损失函数使用。

网络将 VOC anchors 注册为非训练 buffer，默认值来自 `config.yaml`，单位为输出网格 cell。`decode_predictions()` 按 YOLOv2 Region 头解码 raw logits，返回中心点格式的 `boxes_xywh`、`objectness`、`class_probabilities` 和 `class_scores`。默认 box 坐标归一化到输入图像；传入 `normalized=False` 可得到 grid-cell 坐标。解码过程不执行 NMS。

stride=1 的最后一个 max-pool 在右侧和下侧补边，以保持 13×13 网格。stride=2 的池化输入尺寸始终为偶数，因此不需要补边；Darknet 默认参数中对应的尾部补边不会进入有效池化窗口。

类别、输入分辨率和 VOC anchors 初始值见 `config.yaml`。该网络结构与官方 [Darknet YOLOv2-tiny VOC 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg) 对齐。

## 当前接口

```python
from model import YOLOv2Tiny

model = YOLOv2Tiny(num_classes=20, num_anchors=5)
raw_logits = model(images)  # [N, 125, H/32, W/32]
predictions = model.reshape_predictions(raw_logits)
decoded = model.decode_predictions(raw_logits)
```

## 浮点训练和评估

环境依赖为 PyTorch、torchvision、Pillow 和 PyYAML，可使用现有 `pytorch_env`。将 VOC2007 trainval/test、VOC2012 trainval 解压到：

```text
YOLOV2_TINY/data/VOCdevkit/
├── VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   └── ImageSets/Main/
└── VOC2012/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/Main/
```

程序读取本地数据，不自动下载数据集。`config.yaml` 的 `data.root` 可指向已有 VOCdevkit。默认训练使用 2007 train + 2012 train；验证使用两年的 val；最终评估使用 2007 test，划分之间不混用。

在 `YOLOV2_TINY/` 中手动执行：

```bash
python train.py --mode float
python train.py --resume checkpoints/float_model.pth
python evaluate.py --checkpoint checkpoints/float_model_best.pth
```

CUDA 可用时默认使用 `cuda:0`，否则使用 CPU。训练打印每个 epoch 的 loss、验证 mAP50，保存最后一轮 `checkpoints/float_model.pth` 和最佳 `float_model_best.pth`。checkpoint 包含 optimizer、scheduler、epoch 和配置，可恢复训练。当前从随机初始化开始训练。

`dataset.py` 将图像等比例 letterbox 到配置尺寸，框坐标同步缩放和平移，RGB 像素缩放到 `[0,1]`；训练增加随机水平翻转、亮度和颜色增强。标签内部使用归一化 `xyxy + class_id`，保留 difficult 标记。

`loss.py` 实现 YOLOv2 风格损失：按宽高 IoU 选择 anchor，回归 sigmoid 中心偏移和 log 宽高，采用 IoU objectness 目标、重叠负样本忽略以及类别交叉熵。该实现不声称逐项复现 Darknet 的训练数值或 warmup；同一 cell/anchor 冲突时后一个目标覆盖前一个。difficult 对象不参与训练损失。

`evaluate.py` 将 raw logits 解码后执行按类别 NMS（IoU 0.45），按 VOC2007 的 11 点 AP、匹配 IoU 0.5 输出各类 AP 和 mAP。difficult 匹配不计 TP/FP。每图最多保留 300 个检测结果，分数阈值为 0.01。
