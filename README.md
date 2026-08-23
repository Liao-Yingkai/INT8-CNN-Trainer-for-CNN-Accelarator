# MNIST INT8 QAT FPGA 参数导出工程

本项目使用 PyTorch 训练一个轻量级 MNIST CNN，并通过 eager mode QAT（Quantization-Aware Training）获得适合 HLS FPGA CNN Accelerator 的对称 INT8 量化参数。

当前工程覆盖以下流程：

```text
FP32 training
    ↓
PyTorch eager mode QAT
    ↓
INT8 weight / INT32 bias quantization
    ↓
HLS-oriented weight layout conversion
    ↓
NumPy、JSON 和 C header 参数导出
    ↓
Golden integer model validation
```

本仓库只包含 PyTorch 训练、量化、导出和软件验证逻辑，不包含 HLS 内核、片上缓存、systolic array、AXI 接口或 FPGA 工程。

## 模型结构

PyTorch 训练模型使用标准 NCHW 格式，输入 shape 为 `[N, 1, 28, 28]`。导出后的硬件 feature map 使用 NHWC 格式，使卷积矩阵 K 维中的输入通道连续排列。

| 层 | 参数 | 输出 shape |
| --- | --- | --- |
| Input | MNIST 灰度图 | `[N, 1, 28, 28]` |
| Conv1 | `1 → 8`，`3×3`，padding=1 | `[N, 8, 28, 28]` |
| ReLU + MaxPool | `2×2` | `[N, 8, 14, 14]` |
| Conv2 | `8 → 16`，`3×3`，padding=1 | `[N, 16, 14, 14]` |
| ReLU + MaxPool | `2×2` | `[N, 16, 7, 7]` |
| Flatten | `16×7×7` | `[N, 784]` |
| FC | `784 → 10` | `[N, 10]` |

数据预处理采用：

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

## 量化配置

项目使用 `torch.ao.quantization` 提供的 FakeQuant 和 Observer，不包含自定义 FakeQuant 实现。

Activation：

- 数据类型：signed INT8；
- 量化方式：per-tensor symmetric；
- `zero_point = 0`；
- 范围：`[-128, 127]`。

Weight：

- 数据类型：signed INT8；
- 量化方式：per-channel symmetric；
- channel axis：`0`，即输出通道；
- Conv scale 数量等于 `out_channels`；
- Linear scale 数量等于 `out_features`。

Bias：

- 导出数据类型：INT32；
- 每个输出通道独立转换。

核心量化关系为：

```text
weight_int8[channel] = round(weight_fp32[channel] / weight_scale[channel])

bias_int32[channel] = round(
    bias_fp32[channel] / (input_scale × weight_scale[channel])
)
```

层输出 requantization 使用：

```text
real_multiplier[channel]
    = input_scale × weight_scale[channel] / output_scale

real_multiplier ≈ multiplier / 2^shift

output_int8 ≈ (acc_int32 × multiplier) >> shift
```

硬件实现乘法时应使用足够宽的中间数据类型，避免 `INT32 × INT32` 溢出。

## 工程结构

```text
.
├── README.md
└── CNN_FPGA/
    ├── model.py              # CNN 网络和 Conv+ReLU 融合接口
    ├── dataset.py            # MNIST 下载、归一化和 DataLoader
    ├── train.py              # FP32 训练与 QAT 微调入口
    ├── quant.py              # QAT 配置、prepare/load/save 工具
    ├── evaluate.py           # FP32 与 QAT 准确率比较
    ├── export.py             # INT8/INT32 参数和 C header 导出
    ├── golden_model.py       # 导出参数的整数黄金验证模型
    ├── observe_weight.py     # 打印 FP32/QAT checkpoint state_dict
    ├── config.yaml           # 训练超参数
    ├── checkpoints/          # 训练 checkpoint（Git 忽略）
    ├── generated/            # 导出参数（Git 忽略）
    └── data/                 # MNIST 数据集（首次运行生成，Git 忽略）
```

## 环境要求

已验证的开发环境：

- Python 3.10；
- PyTorch 2.6；
- torchvision 0.21；
- NumPy；
- PyYAML。

项目可以使用 CPU，但训练脚本检测到 CUDA 时会自动选择 `cuda:0`。

如果环境中缺少 YAML 支持：

```bash
python -m pip install PyYAML
```

## 配置

训练参数位于 `CNN_FPGA/config.yaml`：

```yaml
data_dir: data
batch_size: 128
num_workers: 4
float_epochs: 10
float_learning_rate: 0.001
qat_epochs: 5
qat_learning_rate: 0.0001
```

QAT 并非从头训练。它必须先加载 `checkpoints/float_model.pth`，再使用更低的学习率继续微调。

## 完整运行流程

所有命令均在 `CNN_FPGA/` 目录运行：

```bash
conda activate pytorch_env
cd CNN_FPGA
```

### 1. FP32 训练

```bash
python train.py --mode float
```

训练过程中打印每个 epoch 的 loss 和 validation accuracy，并生成：

```text
checkpoints/float_model.pth
checkpoints/float_model_best.pth
```

`float_model.pth` 保存最后一个 epoch；`float_model_best.pth` 保存验证准确率最佳的 epoch。

### 2. QAT 微调

```bash
python train.py --mode qat
```

QAT 流程为：

```text
load float_model.pth
→ fuse Conv + ReLU
→ attach FPGA qconfig
→ prepare_qat
→ fine-tune
```

运行结束生成：

```text
checkpoints/qat_model.pth
checkpoints/qat_model_best.pth
```

脚本同时打印 FP32 accuracy、QAT accuracy 和精度差值。

### 3. 评估模型

```bash
python evaluate.py
```

该脚本加载 `float_model.pth` 和 `qat_model.pth`，在 MNIST 测试集上输出：

```text
FP32 accuracy
QAT accuracy
QAT - FP32
```

QAT checkpoint 不能作为普通 `MnistCnn` 直接加载。恢复顺序必须为：

```text
MnistCnn
→ fuse_model
→ create_fpga_qconfig
→ prepare_qat
→ load_state_dict
```

这一过程已封装在 `quant.py` 的 `load_qat_model()` 中。

### 4. 导出 FPGA 参数

```bash
python export.py
```

导出脚本直接读取 QAT FakeQuant/Observer 保存的 scale 和 zero-point，手工生成目标参数；它不依赖 `torch.ao.quantization.convert()` 作为主要转换方式。

生成目录：

```text
generated/
├── weights/
│   ├── conv1_weight_int8.npy
│   ├── conv2_weight_int8.npy
│   └── fc_weight_int8.npy
├── biases/
│   ├── conv1_bias_int32.npy
│   ├── conv2_bias_int32.npy
│   └── fc_bias_int32.npy
├── quant_params/
│   ├── *_weight_scale.npy
│   ├── *_weight_zero_point.npy
│   ├── *_input_scale.npy
│   ├── *_output_scale.npy
│   ├── *_multiplier_int32.npy
│   └── *_shift_int32.npy
├── headers/
│   ├── *_weight.h
│   ├── *_bias.h
│   ├── *_quant.h
│   └── *_shift.h
└── metadata.json
```

`.npy` 文件用于 Python 验证；`.h` 文件可直接作为后续 HLS 工程的参数输入；`metadata.json` 记录 shape、layout、scale、zero-point、bias scale、requant 参数和文件路径。

metadata 当前格式版本为 3。该版本明确区分 PyTorch 的 NCHW 布局与硬件的 NHWC 布局，并记录 FC 权重已从 CHW 输入顺序离线重排为 HWC 输入顺序。

### 5. 黄金模型验证

```bash
python golden_model.py
```

黄金模型：

- 恢复 QAT 模型作为参考；
- 直接解析 `generated/headers/*.h` 中的定点参数；
- 使用 INT8 input/weight、INT32 accumulator/bias 完成整数推理；
- 按层比较 QAT FakeQuant 输出与导出参数推理结果；
- 输出 bit-exact 比例、`±1 LSB` 比例、误差以及 Top-1 accuracy/agreement。

当前脚本固定验证 1000 个 MNIST 测试样本，不需要命令行参数。运行前必须已生成 `qat_model.pth` 和 `generated/metadata.json`/headers。

### 6. 查看 checkpoint 内容（可选）

```bash
python observe_weight.py
```

该脚本打印 `float_model_best.pth` 与 `qat_model_best.pth` 的 `model_state_dict`，主要用于观察 QAT checkpoint 中新增的 FakeQuant/Observer 状态。

## 导出权重布局

PyTorch Conv2d 权重布局为：

```text
[OC, IC, KH, KW]
```

为了适配后续 HLS 矩阵乘法接口，导出时转换为：

```text
[OC, IC, KH, KW]
→ [KH, KW, IC, OC]
→ [IC × KH × KW, OC]
→ [K, OC]
```

各层导出 shape：

| 层 | PyTorch shape | HLS shape |
| --- | --- | --- |
| Conv1 | `[8, 1, 3, 3]` | `[9, 8]` |
| Conv2 | `[16, 8, 3, 3]` | `[72, 16]` |
| FC | `[10, 784]` | `[784, 10]` |

Conv 的 K 维顺序对应 `[KH, KW, IC]`。后续硬件侧生成卷积窗口时必须采用相同的展平顺序。

PyTorch 的 FC 原始权重按照 NCHW feature map 的 CHW flatten 顺序训练：

```text
[OC, C×H×W]
```

FPGA 卷积输出保持 NHWC，直接 flatten 后的输入顺序为 HWC。为了避免硬件运行时执行 NHWC→NCHW feature-map 重排，FC 权重在导出阶段离线转换：

```text
[OC, C×H×W]
→ [OC, C, H, W]
→ [H, W, C, OC]
→ [H×W×C, OC]
```

当前模型对应：

```text
[10, 784]
→ [10, 16, 7, 7]
→ [7, 7, 16, 10]
→ [784, 10]
```

`export.py` 不硬编码 `[16, 7, 7]`。它从最后一个卷积层的 `out_channels` 获取 C，从全连接层的 `in_features` 获取 `C×H×W`，并在当前方形 feature map（`H == W`）约束下使用整数平方根自动推导 H 和 W：

```text
C = conv2.out_channels
H×W = fc.in_features / C
H = W = isqrt(H×W)
```

导出时会检查整除关系和平方关系；如果未来模型使用非方形 feature map，则需要由模型或导出配置显式提供 H/W，不能仅凭 Conv 和 FC 权重唯一推导。

因此硬件可将 `[N, 7, 7, 16]` 的 NHWC 输出直接按 HWC 顺序 flatten 后送入 FC，无需转回 NCHW。FC 导出前后的矩阵 shape 仍是 `[784, 10]`，但 K 维内部的数据顺序已经改变。

## 常见问题

### QAT 模式提示找不到 FP32 checkpoint

必须先执行：

```bash
python train.py --mode float
```

再执行：

```bash
python train.py --mode qat
```

### export 或 golden model 提示缺少文件

严格按照以下顺序运行：

```text
float training → QAT training → export → golden validation
```
