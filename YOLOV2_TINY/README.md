# YOLOv2-tiny VOC

固定硬件目标为 416×416 RGB、VOC 20 类、5 anchors、125 个逻辑输出通道。FP32 保持 Conv+BN+LeakyReLU；QAT 提前折叠 BN，前八层只在 LeakyReLU 后量化，Prediction 无激活。

## 数据准备

在 `YOLOV2_TINY/data/VOCdevkit/` 下准备 `VOC2007/` 和 `VOC2012/`，各包含 `JPEGImages/`、`Annotations/`、`ImageSets/Main/`。训练使用两年的 train，训练期间验证使用两年的 val，最终测试使用 VOC2007 test；脚本不自动下载数据。

数据根目录可在 `config.yaml` 中配置，路径相对 `YOLOV2_TINY/`，与命令启动目录无关。预处理为 RGB、letterbox、像素 `[0,1]`，不是 MNIST mean/std。

## 运行

以下命令从仓库根目录执行：

```text
python YOLOV2_TINY/train.py --mode float
python YOLOV2_TINY/train.py --mode qat --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth
python YOLOV2_TINY/evaluate.py --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth --qat-checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/dump_trace.py --num-samples 1 --seed 1
```

训练和评估支持 `--config`、`--device auto`（默认）、`--device cpu` 或显式设备。QAT 的 `--float-checkpoint` 必须是包含 `model_state_dict` 和 `config` 的未融合 FP32 checkpoint，不能直接传原始 Darknet `.weights`。为保留原有行为，未指定时仍读取 `checkpoints/official_voc_fp32.pth`；该文件不随仓库分发，不会自动下载。

评估应显式选用 QAT 实际初始化的 FP32 和对应最佳 QAT。日志中的 val mAP 与 test mAP 不可混用；外部预训练可能已使用 val，应核实数据来源，不能将其当作独立留出集。

## 导出与验证

- `export.py` 默认读取末轮 `qat_model.pth`，支持 `--checkpoint`、`--output-dir`。只接受 `single_postprocess_v1`；旧双边界 QAT 必须重新微调，不能静默迁移。
- `generated/weights/`、`biases/`、`quant_params/` 保存 NPY；`generated/bins/` 集中保存九层各 weight、bias、quant，共 27 个 BIN。quant 是完整 M 数组后接完整 S 数组。
- `dump_trace.py` 直接读取同批模型 BIN，每张图输出 43 个 BIN 到 `generated/trace/`。PostProcess 包含 bias、LeakyReLU 和 requant，没有独立 bias 中间文件。
- trace 支持 `--generated-dir`、`--output-dir`、`--num-samples`、`--seed`；不会覆盖已有同名文件。更换模型后应重新生成配套 trace。

完整 shape、位序、整数公式与模块比较方法见 [YOLO 详细规范](../docs/yolov2_tiny.md)，硬件数据交接见 [数据说明](../docs/data.md)。

`model.py` 定义网络和解码，`loss.py` 定义检测损失，`quant.py` 管理 QAT；`dataset.py`、`train.py`、`evaluate.py`、`export.py`、`dump_trace.py` 各自保持单一入口。测试集中在 `tests/yolov2_tiny/`：

```text
python scripts/run_tests.py --model yolov2_tiny
```
