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
    ├── dump_trace.py         # 单图模块边界数据导出
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
├── trace/
│   └── trace_data.h
└── metadata.json
```

`.npy` 文件用于 Python 验证；`.h` 文件可直接作为后续 HLS 工程的参数输入；`metadata.json` 记录 shape、layout、scale、zero-point、bias scale、requant 参数和文件路径。

metadata 当前格式版本为 4。该版本明确区分 PyTorch 的 NCHW 布局与硬件的 NHWC 布局，并用 `C,S → S,C` 描述 FC 输入维度的离线重排，其中 `S = H×W`。

### 导出数据说明

`export.py` 生成的数据按“权重、bias、量化参数、C header、元数据”分开保存。数组类型和 shape 如下：

| 路径 | 内容 | 类型 | Conv shape | FC shape |
| --- | --- | --- | --- | --- |
| `weights/*_weight_int8.npy` | 经过 FakeQuant scale 量化并完成 layout 转换的 weight | `numpy.int8` | `[K, OC]` | `[K, OC]` |
| `biases/*_bias_int32.npy` | 按输入 scale 和 weight scale 转换的 bias | `numpy.int32` | `[OC]` | `[OC]` |
| `quant_params/*_weight_scale.npy` | QAT weight FakeQuant 的 per-channel scale | `numpy.float32` | `[OC]` | `[OC]` |
| `quant_params/*_weight_zero_point.npy` | weight zero-point | `numpy.int32` | `[OC]` | `[OC]` |
| `quant_params/*_input_scale.npy` | 当前计算层输入 activation scale | `numpy.float32` | `[1]` | `[1]` |
| `quant_params/*_output_scale.npy` | 当前计算层输出 activation scale | `numpy.float32` | `[1]` | `[1]` |
| `quant_params/*_multiplier_int32.npy` | 每个输出通道的 requant multiplier | `numpy.int32` | `[OC]` | `[OC]` |
| `quant_params/*_shift_int32.npy` | 每个输出通道的 requant shift | `numpy.int32` | `[OC]` | `[OC]` |

其中，`quant_params/` 中的浮点 scale 主要用于 Python 检查和离线分析；FPGA 推理不应读取浮点 scale 来完成 requant，而应使用导出的 `multiplier` 和 `shift`。

`headers/` 保存与上述 NPY 数组内容相同的 C 常量，便于 HLS 工程直接 include：

```cpp
const int8_t  conv1_weight[];
const int32_t conv1_bias[];
const int32_t conv1_multiplier[];
const int32_t conv1_shift[];
```

当前不会生成 scale 的 C header；scale、zero-point 和完整布局信息统一记录在 `metadata.json` 及 `quant_params/` 中。

### `export.py` 的导出逻辑

导出脚本的主要步骤如下：

1. 按 `MnistCnn → fuse_model → create_fpga_qconfig → prepare_qat → load_state_dict` 恢复 QAT 模型。QAT checkpoint 不能直接作为普通 FP32 模型读取。
2. 从每个计算层的 `weight_fake_quant.calculate_qparams()` 读取 per-channel `weight_scale` 和 `weight_zero_point`。
3. 对 FP32 `module.weight` 按输出通道执行：

   ```text
   weight_int8[channel] = round(weight_fp32[channel] / weight_scale[channel])
   ```

   结果夹紧到 `[-128,127]` 后保存为 INT8。由于配置为 symmetric，weight zero-point 应为 0。
4. 转换 weight layout：

   - Conv：`[OC,IC,KH,KW] → [KH,KW,IC,OC] → [KH×KW×IC,OC]`；
   - FC：原始 PyTorch FC 输入顺序是 `[C,S]`，其中 `S=H×W`；硬件 NHWC flatten 顺序是 `[S,C]`，因此执行 `[OC,C,S] → [S,C,OC] → [S×C,OC]`。

   FC 的通道数 C 从最后一个 Conv 的 `out_channels` 自动获取，空间元素数由 `fc.in_features / C` 得到，不需要单独推断 H 和 W。
5. 从 activation FakeQuant 读取 per-tensor activation scale。第一层使用模型输入 scale，后续层使用前一层输出 scale；activation zero-point 为 0。
6. 按输出通道把 bias 转换为 INT32：

   ```text
   bias_int32[channel] = round(
       bias_fp32[channel] /
       (input_scale × weight_scale[channel])
   )
   ```

7. 计算每个输出通道的真实 requant 比例，并编码为整数 multiplier/shift：

   ```text
   real_multiplier[channel]
       = input_scale × weight_scale[channel] / output_scale

   real_multiplier ≈ multiplier[channel] / 2^shift[channel]
   ```

   HLS 侧可使用等价的整数形式：

   ```text
   output_int8 ≈ (acc_int32 × multiplier) >> shift
   ```

8. 将 NPY、C header 和 `metadata.json` 写入 `generated/`，并在终端打印每层原始 weight shape、导出 shape、scale 数量和 bias 数量。

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

### 6. 导出单图模块边界数据

```bash
python dump_trace.py
```

脚本会从 MNIST 测试集中随机选择一张图片；如果本地没有测试集，会自动下载到 `CNN_FPGA/data/`。脚本使用与训练相同的 `ToTensor()` 和 Normalize，并调用 `golden_model.py` 中已有的定点推理逻辑，不重新实现卷积。

默认使用 `seed=1` 选择测试图片。如需选择其他可复现样本，可以指定随机种子：

```bash
python dump_trace.py --seed 123
```

运行时会打印所选测试样本的 index 和 label。

输出文件为：

```text
generated/trace/trace_data.h
```

该 header 可以直接被 HLS CSim include。它包含以下模块边界数据：

| 数组 | 类型 | shape | 语义 |
| --- | --- | --- | --- |
| `trace_input_int8_hwc` | `int8_t` | `[28,28,1]` | 量化后的输入 |
| `trace_conv1_unfold_int8` | `int8_t` | `[784,9]` | Conv1 的 im2row/unfold 输出 |
| `trace_conv1_systolic_int32_nhwc` | `int32_t` | `[28,28,8]` | Conv1 MAC，未加 bias |
| `trace_conv1_postprocess_int32_nhwc` | `int32_t` | `[28,28,8]` | Conv1 加 bias、ReLU 后 |
| `trace_conv1_requant_int8_nhwc` | `int8_t` | `[28,28,8]` | Conv1 requant 后 |
| `trace_pool1_int8_nhwc` | `int8_t` | `[14,14,8]` | MaxPool1 后 |
| `trace_conv2_unfold_int8` | `int8_t` | `[196,72]` | Conv2 的 im2row/unfold 输出 |
| `trace_conv2_systolic_int32_nhwc` | `int32_t` | `[14,14,16]` | Conv2 MAC，未加 bias |
| `trace_conv2_postprocess_int32_nhwc` | `int32_t` | `[14,14,16]` | Conv2 加 bias、ReLU 后 |
| `trace_conv2_requant_int8_nhwc` | `int8_t` | `[14,14,16]` | Conv2 requant 后 |
| `trace_pool2_int8_nhwc` | `int8_t` | `[7,7,16]` | MaxPool2 后 |
| `trace_fc_systolic_int32` | `int32_t` | `[10]` | FC MAC，未加 bias |
| `trace_fc_postprocess_int32` | `int32_t` | `[10]` | FC 加 bias 后 |
| `trace_fc_requant_int8` | `int8_t` | `[10]` | FC requant 后 |

Conv 的 postprocess 输出包含 ReLU。当前模型 FC 后没有 ReLU，因此 FC postprocess 只包含 bias 加法。systolic 和 postprocess 阶段本质上是 INT32，requant 和池化输出是 INT8；脚本会在写出前检查每个数组的 dtype 和取值范围。

两个 `unfold` 数组是卷积 systolic 输入的直接参考数据，采用 `[OH×OW, KH×KW×Cin]` 排布。每一行对应一个输出像素，列顺序为 `[KH, KW, Cin]`，与导出的卷积权重 `[KH, KW, Cin, Cout] → [K, Cout]` 相匹配；padding=1 的边界位置使用量化零值填充。

HLS CSim 中可直接使用：

```cpp
#include "trace_data.h"

// 例如比较 Conv1 requant 输出
if (conv1_output[y][x][c] != trace_conv1_requant_int8_nhwc[(y * 28 + x) * 8 + c]) {
    // report mismatch
}
```

该 trace 文件不包含额外 JSON metadata、scale 或浮点数据；shape 通过 C 数组声明和 header 注释给出，量化参数继续使用 `generated/headers/` 中的导出参数。

### 7. 查看 checkpoint 内容（可选）

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

PyTorch 的 FC 原始权重按照 NCHW feature map 的 CHW flatten 顺序训练。由于 NCHW→NHWC 不改变 H、W 的相对顺序，可以将两个空间维合并为 `S = H×W`：

```text
[OC, C×S]
```

FPGA 卷积输出保持 NHWC，直接 flatten 后的输入顺序为 `[S,C]`。为了避免硬件运行时执行 NHWC→NCHW feature-map 重排，FC 权重在导出阶段离线转换：

```text
[OC, C×S]
→ [OC, C, S]
→ [S, C, OC]
→ [S×C, OC]
```

当前模型对应：

```text
[10, 784]
→ [10, 16, 49]
→ [49, 16, 10]
→ [784, 10]
```

`export.py` 从最后一个卷积层的 `out_channels` 自动获取 C，并根据 FC 权重的 `input_features` 计算空间元素数量 S：

```text
C = conv2.out_channels
S = fc.in_features / C
```

导出时只需检查 `fc.in_features` 能否被 C 整除，不需要推导独立的 H、W，也不要求 feature map 为方形。

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
