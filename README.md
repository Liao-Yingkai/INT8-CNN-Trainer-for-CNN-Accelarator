# INT8 CNN FPGA 软件参考工程

用于 FPGA CNN 加速器的 PyTorch 软件部分：FP32 训练、官方 eager-mode QAT、整数参数导出和模块边界参考数据。包含 MNIST CNN 与 YOLOv2-tiny VOC 两条独立流程，不包含 HLS、板级工程或硬件构建工具。

完整流程为：**数据准备 → FP32 训练 → QAT 微调 → 模型评估 → 参数导出 → 整数参考与 trace → 硬件端比较**。也可以直接使用另行提供的 checkpoint 或导出包，从其中一个阶段开始，不要求每次重新训练。

[运行流程](#快速开始) · [数据准备](#数据准备) · [使用已训练模型](#使用另行提供的模型或导出包) · [导出数据说明](#导出数据有哪些如何使用) · [测试](#测试)

## 支持的模型

| 模型 | 输入与任务 | 网络概要 | 硬件导出 |
|---|---|---|---|
| MNIST | 28×28×1 灰度，10 类分类 | 两层 Conv+ReLU+Pool，FC 784→10 | NPY、C header、metadata、逐图 trace header |
| YOLOv2-tiny | 416×416×3 RGB，VOC 20 类检测 | 八层 Conv+LeakyReLU、六层 Pool、1×1 Prediction | NPY、27 个模型 BIN、metadata、每图 43 个 trace BIN |

两者均使用官方 `torch.ao.quantization` eager-mode QAT：activation 为 per-tensor symmetric signed INT8，weight 为 OC 轴 per-channel symmetric signed INT8，zero_point=0。QAT checkpoint 仍保存浮点可训练权重和 FakeQuant 状态；真正的 INT8 权重、INT32 bias 和整数 multiplier/shift 由导出器生成。

这是针对固定加速器拓扑的参考软件，不是任意模型的通用部署框架。MNIST 和 YOLO 的产物格式、metadata 版本及预处理不同，不能互换。

## 仓库结构

```text
.
├── MNIST/                # MNIST 源码、配置与使用入口
├── YOLOV2_TINY/           # YOLOv2-tiny 源码、配置与使用入口
├── docs/                 # 网络、量化、导出 ABI 与数据交接规范
├── scripts/              # 统一测试和 checkpoint 检查工具
├── tests/                # 按模型隔离的无数据集回归测试
├── requirements.txt      # 依赖基线
└── CONTRIBUTING.md       # 路径、测试和提交约定
```

模型目录内保留 `data/`、`checkpoints/`、`generated/` 作为运行产物位置，不随 Git 分发。现有目录名及导出文件名保持兼容；没有把所有模型包装成通用推理框架。

## 快速开始

要求 Python 3.10 或以上，参考依赖使用 PyTorch 2.6 / torchvision 0.21。CPU 可完成测试、导出和推理；训练可选 CUDA，不依赖特定设备。安装依赖：

```text
python -m pip install -r requirements.txt
```

解释器版本还需符合所选依赖包的支持范围；依赖清单不指定 CUDA 构建来源，使用加速设备时应选择版本配套的 PyTorch/torchvision 构建。

以下命令以仓库根目录为工作目录：

```text
python MNIST/train.py --mode float
python MNIST/train.py --mode qat
python MNIST/evaluate.py
python MNIST/export.py
python MNIST/golden_model.py
python MNIST/dump_trace.py --num-samples 1 --seed 1
```

MNIST 的 QAT 默认从 `MNIST/checkpoints/float_model.pth` 初始化；评估默认比较末轮 FP32 和 QAT。`golden_model.py` 比较 QAT 与导出 header 驱动的整数参考；`dump_trace.py` 抽取测试图并导出模块边界 header。它们需要已准备好的对应文件，不会自动替用户完成训练。

YOLO 在准备好 VOC 数据后执行：

```text
python YOLOV2_TINY/train.py --mode float
python YOLOV2_TINY/train.py --mode qat --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth
python YOLOV2_TINY/evaluate.py --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth --qat-checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/dump_trace.py --num-samples 1 --seed 1
```

上面的 YOLO 示例显式使用最佳 FP32/QAT；如果直接执行 `python YOLOV2_TINY/train.py --mode qat`，默认初始化文件是 `YOLOV2_TINY/checkpoints/float_model.pth`。默认 `evaluate.py` 和 `export.py` 同样选择各自末轮文件，不会自动改选 `*_best.pth`。使用外部 FP32 初始化 QAT 时，精度比较也应指定该 FP32，避免把不相关的自训练模型当作量化前基线。

YOLO QAT 方案为 `single_postprocess_v1`：前八层取消 LeakyReLU 前的 activation FakeQuant，保留其后的量化边界；Prediction 不加 LeakyReLU。旧双边界 QAT checkpoint 会明确拒绝，不能只改文件名后继续使用。

### 配置、设备与输出

```text
python MNIST/train.py --mode float --device cpu --config MNIST/config.yaml
python YOLOV2_TINY/train.py --mode qat --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth --device auto
```

`config.yaml` 管理批大小、数据位置、训练轮数和学习率等；QAT 使用更低学习率微调，不是重新随机训练。`--device auto` 自动选设备，也可显式选择 CPU 或其他可用设备。配置和设备参数用于训练/评估，不代表所有脚本都有相同参数；可通过相应入口的 `--help` 查看支持项。

训练产物命名统一：

| 文件 | 含义 |
|---|---|
| `checkpoints/float_model.pth` | 最后一轮 FP32 |
| `checkpoints/float_model_best.pth` | 按训练期间验证指标选择的最佳 FP32 |
| `checkpoints/qat_model.pth` | 最后一轮 QAT |
| `checkpoints/qat_model_best.pth` | 按训练期间验证指标选择的最佳 QAT |

MNIST 报告分类 accuracy；YOLO 报告 IoU=0.5、VOC2007 11 点 AP 的 mAP50。训练阶段的验证集与最终测试集用途不同，不能直接混用指标；外部预训练是否使用过验证集，应以其训练说明为准。最佳模型也不等于已经通过整数硬件精度验证。

脚本不会因导入而训练。训练会写入固定 checkpoint 名称，运行前保留需要的已有结果。默认路径按源码位置定位；显式 CLI 相对路径按启动目录解释；配置里的数据路径相对所属模型目录，详见[开发约定](CONTRIBUTING.md)。

## 数据准备

### MNIST

默认数据根目录为 `MNIST/data/`。数据加载器可自动下载 MNIST；离线使用时应预先准备 torchvision 可识别的文件，典型结构为：

```text
MNIST/data/MNIST/raw/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

不要只把压缩文件放到任意目录后假设程序会找到它们。输入预处理为 `ToTensor()` 和 `Normalize((0.1307,), (0.3081,))`，PyTorch 计算使用 NCHW。当前实现使用官方 train 训练，官方 test 在训练期间报告准确率，没有另行划分独立 validation；该指标应按实际划分解释。

### PASCAL VOC

YOLO 不自动下载 VOC。训练和训练期间验证需要 VOC2007 trainval、VOC2012 trainval；最终测试和 trace 抽样需要 VOC2007 test。解压到：

```text
YOLOV2_TINY/data/VOCdevkit/
├── VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   └── ImageSets/Main/    # train.txt、val.txt、test.txt
└── VOC2012/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/Main/    # train.txt、val.txt
```

只训练时不要求 test；只评估或导出 trace 时不要求 trainval。YOLO 预处理为 RGB、等比例 letterbox 到 416×416、像素缩放到 `[0,1]`，不应用 MNIST mean/std。修改训练数据位置使用 `YOLOV2_TINY/config.yaml` 的 `data.root`；trace 入口也读取该默认配置的数据位置，不自动继承另一次训练命令的 `--config`。

## 使用另行提供的模型或导出包

**源码仓库不提交 `.pth`、`.bin`、数据集或生成的参数，但这些仍是完整工作流的一部分。** 后续可能通过独立附件或发布包提供训练模型、导出参数及参考数据；具体下载位置、版本、文件范围以实际发布说明为准。目前这里不假设存在可下载的数据包，也不自动下载模型。

### 先确定拿到的是哪种数据

| 已有内容 | 可以直接做什么 | 仍需准备什么 |
|---|---|---|
| 只有源码 | 运行合成回归测试、检查代码和格式 | 训练/评估需要数据，导出需要 QAT checkpoint |
| 兼容 FP32 `.pth` | 初始化 QAT；有测试数据时评估 FP32 | QAT 微调需要训练与验证数据 |
| 兼容 QAT `.pth` | 不重新训练即可导出参数 | 评估需要测试数据；重新导出 trace 也需要对应测试图 |
| YOLO `metadata.json` + 27 个模型 BIN | 交给硬件端使用；有 VOC test 时生成新 trace | 不能仅凭 BIN 直接继续 PyTorch 训练 |
| 同批模型 BIN + 完整 trace | 硬件模块/整网参考比较 | 不要求 checkpoint 或完整训练数据集；硬件构建条件由加速器仓库说明 |

`.pth` 必须符合本工程 checkpoint 结构；原始 Darknet `.weights`、裸 `state_dict` 或其他框架文件不是可直接替换的输入。只加载可信来源的 checkpoint，并核对包内的网络配置、QAT 方案、预处理和校验标识。

### 使用提供的 checkpoint

将模型放到对应模型目录的 `checkpoints/`，或者使用受支持的 CLI 参数选择文件，不需要重新执行 FP32 训练。例如只有 YOLO QAT 时：

```text
python YOLOV2_TINY/evaluate.py --mode qat --qat-checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth --output-dir YOLOV2_TINY/generated/provided_model
python YOLOV2_TINY/dump_trace.py --generated-dir YOLOV2_TINY/generated/provided_model --num-samples 1 --seed 1
```

其中第一步评估和最后一步 trace 生成需要 VOC2007 test，参数导出不需要数据集。

MNIST 的评估可用 `--float-checkpoint`、`--qat-checkpoint` 选择文件；但当前 MNIST 的 export/golden 入口固定读取 `MNIST/checkpoints/qat_model.pth`。如果提供的是 `qat_model_best.pth`，希望导出它，应先保留原文件，再把选中的模型复制为该默认文件名。不要给 MNIST exporter 传不存在的 `--checkpoint` 参数。

### 使用已导出的数据

如数据包已包含 `bins/`、`metadata.json` 和 `trace/`，不必重新训练或导出；按其原目录关系放置，例如：

```text
YOLOV2_TINY/generated/provided_model/
├── metadata.json
├── bins/                 # 27 个模型 BIN
└── trace/                # 可选，同一模型的单图或多图参考 BIN
```

不要多嵌套一层包名或 `generated/` 后又使用错误的 `--generated-dir`。该参数应指向直接包含 metadata 和 bins 的目录。有完整 trace 时可以直接交给硬件端比较，不需要再次运行 `dump_trace.py`；只有 BIN、没有 metadata 时，不足以让本仓库的 trace 脚本恢复输入 scale 和格式说明。

建议另行发布的数据包至少说明：对应源码版本、模型配置、数据划分、指标与评估口径、初始化来源、checkpoint 文件及 SHA256、导出版本和使用命令。若包含 trace，还应记录样本数、seed、VOC 图像 ID/索引或等效的样本标识。不要把尚未执行的整数精度或硬件验证写成已通过；发布建议见[数据说明](docs/data.md#独立数据包的建议内容)。

## 导出数据有哪些，如何使用

### YOLO 模型参数

```text
YOLOV2_TINY/generated/
├── metadata.json
├── weights/              # 逻辑 INT8 weight NPY
├── biases/               # INT32 bias NPY
├── quant_params/         # scale、zero_point、M/S NPY
├── bins/
│   ├── conv1_weight.bin
│   ├── conv1_bias.bin
│   ├── conv1_quant.bin
│   ├── ...               # conv2～conv8，每层三份
│   ├── prediction_weight.bin
│   ├── prediction_bias.bin
│   └── prediction_quant.bin
└── trace/                # 逐图模块参考，不是模型参数
```

| 数据 | 类型与排列 | 使用方 |
|---|---|---|
| weight BIN | INT8；`[ceil(Cout/4),Kh*Kw*ceil(Cin/8),4,8]`，尾通道填零 | 硬件权重加载与 Systolic |
| bias BIN | little-endian INT32，逻辑 `[Cout]` | PostProcess |
| quant BIN | little-endian INT32，完整 multiplier `[Cout]` 后接完整 shift `[Cout]` | PostProcess |
| scale / zero_point NPY | scale 为浮点、zero_point 为整数 | 软件检查和离线分析，不用于硬件浮点 requant |
| metadata | 网络配置、量化方案、shape、scale、M/S、相对文件路径及字节数 | 格式检查、软件读取及硬件侧配置信息生成 |

每层只有一组 bias/M/S；bias、LeakyReLU、requant 在同一个 PostProcess 中完成，Prediction 禁用 LeakyReLU。当前导出使用 metadata v2、权重布局 v3；QAT 方案标记是 `single_postprocess_v1`。参数数量按逻辑 Cout，不因权重补齐而扩充，Prediction 的 bias 和 quant 分别为 500、1000 字节。完整整数公式及 BIN 位序见 [YOLO 规范](docs/yolov2_tiny.md)。

### 模块参考数据 trace

YOLO 每张测试图生成 43 个 BIN：输入 INT8 HWC，九层各自的逻辑 unfold、打包 im2row、Systolic INT32 HWC、PostProcess INT8 HWC，以及六层 Pool INT8 HWC。文件名末尾 `_0000`、`_0001` 表示导出顺序，不是数据集索引；样本 ID 在终端打印。

```text
python YOLOV2_TINY/dump_trace.py --num-samples 3 --seed 1 --output-dir YOLOV2_TINY/generated/trace_batch3
```

trace 直接读取导出的模型 BIN 做数学参考，不是把 QAT 浮点输出改成整数类型；也不模拟 systolic 周期、FIFO 或反压。Systolic 输出不含 bias，PostProcess 输出已经包含全部后处理。无效 Pack Lane 的补零与 HWC 的逻辑数据必须按各自 shape 比较，不能用相同平坦下标混读。

MNIST 使用 `generated/headers/` 和 `generated/trace/trace_data_0000.h`，不是上述 YOLO BIN 格式。其 header 由 `golden_model.py` 读取做整数参考，FC 权重已按 HWC flatten 顺序离线重排，详见 [MNIST 规范](docs/mnist.md)。

### 交给 FPGA 工程

YOLO 同一次导出的 `bins/`、`trace/` 和 metadata 分别对应加速器仓库的 `data/yolov2_tiny/bins/`、`data/yolov2_tiny/trace/`、`data/yolov2_tiny/metadata.json`。硬件测试只需要其测试项使用的模型与参考 BIN，不需要训练 checkpoint 或完整数据集。

软件 trace 的 Prediction 是紧凑 `13×13×125`；硬件 DDR 可能补齐通道步长，不能直接按文件长度或字节偏移混读。SD 卡文件组织、硬件配置生成和运行方式由硬件仓库维护，本软件不会自动修改另一仓库。详细交接见 [数据说明](docs/data.md)。

### 保持数据一致

- checkpoint、模型 BIN、输入/输出 scale、anchors、metadata 和 trace 必须配套；更换权重后重新导出整套参数与参考。
- 训练和参数导出会写入固定或指定目标，先保留需要的旧结果。不同实验建议用不同的 `generated/` 子目录。
- YOLO trace 拒绝覆盖已有同名文件；MNIST trace 会清理同目录中旧的匹配 trace header，再生成新的一批，运行前应保存需要的结果。
- NPY 是便于软件验证的数组文件，不能直接当作无文件头的 BIN；`.pth` 也不能直接交给 FPGA 当作 INT8 权重读取。
- 文件长度、版本或 QAT 方案不匹配时，先核对数据包和代码版本，不要删除校验或用 `strict=False` 强行加载。

## 使用与数据契约

| 入口 | 内容 |
|---|---|
| [MNIST](MNIST/README.md) | 28×28 灰度分类训练、QAT、header 导出与黄金验证 |
| [YOLOv2-tiny](YOLOV2_TINY/README.md) | 416×416 RGB VOC 检测、QAT、27 个模型 BIN、43 个单图 trace BIN |
| [MNIST 导出规范](docs/mnist.md) | NCHW/HWC、FC 离线重排、NPY/header 及整数公式 |
| [YOLO 导出规范](docs/yolov2_tiny.md) | 单次 PostProcess、权重 Pack、BIN 位序、metadata 和 trace |
| [数据与硬件交接](docs/data.md) | 数据集、checkpoint 格式、产物边界及 HLS 数据目录对应关系 |

仓库不附带训练 checkpoint、数据集或生成的模型文件，也不假设存在某个外部目录。仅克隆源码可以运行合成回归测试，但不能直接完成需要已训练参数的导出或仿真。YOLO 不自动下载 VOC 或官方权重；使用外部 checkpoint 时需自行取得符合接口的文件及其使用许可。

## 测试

```text
python scripts/run_tests.py
python scripts/run_tests.py --model mnist
python scripts/run_tests.py --model yolov2_tiny
python scripts/check_repository.py
python scripts/inspect_checkpoint.py MNIST/checkpoints/float_model_best.pth
```

回归使用临时合成数据，不访问训练数据集、不自动训练、不覆盖用户 checkpoint。测试通过表示软件数值和文件契约通过对应检查，不等同于检测精度、HLS 仿真、综合或板测通过。
