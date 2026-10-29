# YOLOv2-tiny VOC PyTorch 工程

当前阶段包含网络定义。VOC 数据加载、检测损失、训练、评估、QAT 和 FPGA 参数导出会在后续阶段逐步加入；MNIST 工程仍位于 `CNN_FPGA/`，两者相互独立。

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

前 8 个卷积均为 Conv + BatchNorm + LeakyReLU（斜率 `0.1`）；prediction 卷积不带 BatchNorm 和激活函数，返回 raw logits。VOC 检测头默认 `5 × (5 + 20) = 125` 个通道。模型输入为 PyTorch 标准 NCHW；输入高宽须是 32 的倍数。`reshape_predictions()` 可将输出整理为 `[N, anchors, grid_h, grid_w, 5 + classes]`，不执行 sigmoid、box decode 或 NMS。

stride=1 的最后一个 max-pool 在右侧和下侧补边，以保持 13×13 网格。stride=2 的池化输入尺寸始终为偶数，因此不需要补边；Darknet 默认参数中对应的尾部补边不会进入有效池化窗口。

类别、输入分辨率和 VOC anchors 初始值见 `config.yaml`。该网络结构与官方 [Darknet YOLOv2-tiny VOC 配置](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg) 对齐。

## 当前接口

```python
from model import YOLOv2Tiny

model = YOLOv2Tiny(num_classes=20, num_anchors=5)
raw_logits = model(images)  # [N, 125, H/32, W/32]
predictions = model.reshape_predictions(raw_logits)
```
