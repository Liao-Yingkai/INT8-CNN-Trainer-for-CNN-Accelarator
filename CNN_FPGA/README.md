# MNIST CNN

28×28×1 输入，Conv(1→8) → ReLU → Pool → Conv(8→16) → ReLU → Pool → Flatten → FC(784→10)。PyTorch 使用 NCHW；硬件参考使用 HWC/NHWC，FC 权重在导出时离线重排。

## 运行

以下命令从仓库根目录执行，依赖见 [根说明](../README.md)。

```text
python CNN_FPGA/train.py --mode float
python CNN_FPGA/train.py --mode qat
python CNN_FPGA/evaluate.py
python CNN_FPGA/export.py
python CNN_FPGA/golden_model.py
python CNN_FPGA/dump_trace.py --num-samples 1 --seed 1
```

训练支持 `--config CNN_FPGA/config.yaml`、`--device auto`（默认）或 `--device cpu`，QAT 可用 `--float-checkpoint` 选择初始化文件。没有显式初始化时仍读取 `CNN_FPGA/checkpoints/float_model.pth`。首次使用时数据加载器可下载 MNIST；在离线条件下应预先准备数据。

## 文件职责

| 文件 | 职责 |
|---|---|
| `model.py` / `quant.py` | 网络、Conv+ReLU 融合、官方 FakeQuant 和 checkpoint |
| `dataset.py` / `config.yaml` | MNIST 归一化、数据加载及训练参数 |
| `train.py` / `evaluate.py` | FP32/QAT 训练和分类准确率评估 |
| `export.py` | INT8 权重、INT32 bias/M/S、NPY、C header 和 metadata |
| `golden_model.py` / `dump_trace.py` | header 驱动整数参考及测试图模块边界 header |

## 运行产物

`data/` 保存数据；`checkpoints/` 保存末轮和最佳 FP32/QAT；`generated/` 保存参数与 trace。这些目录不会随 Git 提供，数值格式见 [MNIST 详细规范](../docs/mnist.md)。不要与 YOLO 的 BIN/metadata 版本混用。

测试入口：`python scripts/run_tests.py --model mnist`。只检查 checkpoint 内容时使用 `python scripts/inspect_checkpoint.py CNN_FPGA/checkpoints/qat_model_best.pth`，不会恢复网络或启动训练。
