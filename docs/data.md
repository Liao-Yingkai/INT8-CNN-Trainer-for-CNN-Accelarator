# 数据、checkpoint 与硬件交接

所有路径相对本仓库根目录。数据和生成物不随源码克隆提供，脚本不会假设已有个人目录或其他工程。

## 输入与 checkpoint

| 模型 | 数据位置 | QAT 初始化 |
|---|---|---|
| MNIST | `MNIST/data/` | 默认 `MNIST/checkpoints/float_model.pth`，可用训练参数覆盖 |
| YOLO | `YOLOV2_TINY/data/VOCdevkit/` | 默认 `YOLOV2_TINY/checkpoints/float_model.pth`，可用 `--float-checkpoint` 覆盖 |

MNIST 加载器支持下载；YOLO 读取预先准备的 VOC2007 trainval/test 和 VOC2012 trainval。checkpoint 由 PyTorch 保存，MNIST 包含 `model_state_dict`、epoch、accuracy；YOLO 还包含模型构造 `config` 和 mAP，QAT 包含明确方案标记。原始 Darknet `.weights`、其他框架文件不能直接当成本工程 checkpoint。

只加载可信来源的 checkpoint。哈希用于标识文件，不构成真实性或安全性保证。旧模型迁移需明确说明，不能用 `strict=False` 掩盖量化边界不匹配。

## 导出产物

| 模型 | 产物 | 规范 |
|---|---|---|
| MNIST | NPY、metadata v5、布局 v2 的 header、逐图 trace header | [MNIST 规范](mnist.md) |
| YOLO | NPY、metadata v2、权重布局 v3、27 个模型 BIN、每图 43 个 trace BIN | [YOLO 规范](yolov2_tiny.md) |

参数与 trace 必须来自同一个 checkpoint。更换模型后重新生成整套数据，不混用输入 scale、anchors、权重、bias/M/S 和输出参考。YOLO metadata 中的 checkpoint 路径只作为相对来源标识，SHA256 用于标识实际文件；读取方不应依赖训练时的文件位置。MNIST metadata 不包含相同的来源哈希字段，发布时可通过外部清单补充，不应假设两个 metadata schema 一致。

## 独立数据包的建议内容

源码不跟踪 `.pth`、模型 BIN 和 trace，不表示使用者必须自己重新训练。维护者可另行提供 checkpoint 包、导出包或二者合并的包；具体内容和下载方式以实际发布说明为准。建议保留与源码相同的相对目录，例如：

```text
model_package/
├── README.md                  # 本包的来源、用法、版本和验证范围
├── SHA256SUMS                 # 建议提供的完整文件校验清单
├── MNIST/                     # 按发布范围可选
│   ├── config.yaml
│   ├── checkpoints/           # FP32/QAT，注明末轮或最佳
│   └── generated/             # metadata、headers、可选 NPY 和 trace
└── YOLOV2_TINY/                # 按发布范围可选
    ├── config.yaml
    ├── checkpoints/           # 可选，训练/评估/重新导出时使用
    └── generated/
        ├── metadata.json
        ├── bins/              # 一套模型完整的 27 个 BIN
        └── trace/             # 可选，每张图完整 43 个参考 BIN
```

这是建议的发布组织，不代表仓库已提供该包。可以只发布其中一种模型，或分别发布 checkpoint 与导出包，但需记录其对应关系。NPY 适合软件调试，可作为可选附加文件；YOLO 的硬件模型加载不需要 NPY，也不需要 checkpoint。

包内说明至少应包括：

| 信息 | 应说明的内容 |
|---|---|
| 源码版本 | 对应提交或发布版本，不仅写“最新” |
| 模型与预处理 | 模型名、输入尺寸/通道、类别顺序、anchors、归一化或 letterbox |
| 初始化与训练数据 | 从零或预训练初始化、数据集和 train/val/test 划分、外部权重来源及许可 |
| checkpoint | 文件名、FP32/QAT 类型、末轮/最佳、epoch、初始化对应关系、文件 SHA256 |
| 量化与导出 | QAT 方案、metadata/layout 版本、bias/M/S 和权重文件是否属于同一导出 |
| 精度与验证 | accuracy/mAP 的数据划分、阈值和 AP 口径；FakeQuant/整数参考/硬件结果分别报告 |
| trace | 样本数量、seed、图像 ID 或索引、文件后缀含义，以及对应的参数包 |
| 使用方法 | 哪些文件放在哪里、可跳过哪些训练步骤、复现命令和需要的数据集 |

源码版本与数据包版本不必使用相同名称，但必须有可查的对应关系。校验和用于发现下载损坏和版本混用，不能替代来源可信性或使用许可检查。

## 接收数据包后的操作

1. 先阅读包内说明，核对格式和校验清单，不直接覆盖已有实验。
2. checkpoint 放到相应模型的 `checkpoints/`，或用 CLI 显式选择；配置文件先比较差异，不盲目替换自定义配置。
3. 已导出的 YOLO 数据可放到独立的 `YOLOV2_TINY/generated/<实验名>/`，该目录应直接包含 `metadata.json` 和 `bins/`。使用它生成 trace 时把该目录作为 `--generated-dir`。
4. 有兼容 QAT checkpoint 时可以直接运行 exporter，不需要数据集；评估或生成新 trace 时仍需测试数据。只有模型 BIN 时不能恢复为本工程可继续训练的 QAT checkpoint。
5. 有同批 BIN 和完整 trace 时，硬件比较可直接使用，不需要重复训练、重新导出或取得完整训练集。

MNIST 当前 exporter/golden 使用固定默认路径：checkpoint 为 `MNIST/checkpoints/qat_model.pth`，产物为 `MNIST/generated/`。若单独提供最佳 checkpoint，可在保留旧文件后复制为该默认名；若提供完整导出包，将其目录内容对应到该默认 generated 目录。QAT 对照验证还需要对应的 QAT checkpoint 和 MNIST 测试数据，不能仅靠 header 做 QAT 精度比较。

YOLO 支持独立的 checkpoint 和导出目录。比如先将提供的模型放入相应目录，再执行以下可选步骤：

```text
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth --output-dir YOLOV2_TINY/generated/provided_model
python YOLOV2_TINY/dump_trace.py --generated-dir YOLOV2_TINY/generated/provided_model --output-dir YOLOV2_TINY/generated/provided_model/trace --num-samples 1 --seed 1
```

已经拿到完整导出包时不要重复导出；已有同名 trace 时脚本会拒绝覆盖，应另选输出目录。不要把某个包的 BIN 与另一个包的 metadata 或 trace 合并。数据集、checkpoint、模型 BIN 和 trace 都有不同用途，不因为扩展名相同就具有可替换性。

## 与加速器仓库对接

软件与硬件是独立仓库，不要求特定相邻目录名。将 YOLO 同一次导出的以下内容放入加速器仓库约定位置：

| 软件产物 | 加速器仓库中的目标位置 |
|---|---|
| `YOLOV2_TINY/generated/bins/` 中的 27 个文件 | `data/yolov2_tiny/bins/` |
| `YOLOV2_TINY/generated/trace/` 中的配套 trace | `data/yolov2_tiny/trace/` |
| `YOLOV2_TINY/generated/metadata.json` | `data/yolov2_tiny/metadata.json` |

复制目录内容，避免再嵌套一层 `generated/` 或 `bins/`。硬件的 SD 卡、DDR 写回步长和 PS 配置生成流程以其自身文档为准：软件 trace 始终保存逻辑 HWC，Prediction 是紧凑 `13×13×125`，不能假定与硬件补齐后的存储步长相同。

这里保持文件名和 ABI，不在软件工程内写入硬件工程路径或调用硬件工具。模块级比较的数据需求见 YOLO 规范，不需要将训练数据集交给硬件仿真。
