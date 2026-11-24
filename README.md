# INT8 CNN FPGA 软件参考工程

用于 FPGA CNN 加速器的 PyTorch 软件部分：FP32 训练、官方 eager-mode QAT、整数参数导出和模块边界参考数据。包含 MNIST CNN 与 YOLOv2-tiny VOC 两条独立流程，不包含 HLS、板级工程或硬件构建工具。

## 仓库结构

```text
.
├── MNIST/              # MNIST 源码、配置与使用入口
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
```

YOLO 在准备好 VOC 数据后执行：

```text
python YOLOV2_TINY/train.py --mode float
python YOLOV2_TINY/train.py --mode qat --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth
python YOLOV2_TINY/evaluate.py --float-checkpoint YOLOV2_TINY/checkpoints/float_model_best.pth --qat-checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/dump_trace.py --num-samples 1 --seed 1
```

脚本不会因导入而训练。训练会写入固定 checkpoint 名称，运行前保留需要的已有结果。默认路径按源码位置定位；显式 CLI 相对路径按启动目录解释；配置里的数据路径相对所属模型目录，详见[开发约定](CONTRIBUTING.md)。

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
python scripts/inspect_checkpoint.py MNIST/checkpoints/float_model_best.pth
```

回归使用临时合成数据，不访问训练数据集、不自动训练、不覆盖用户 checkpoint。测试通过表示软件数值和文件契约通过对应检查，不等同于检测精度、HLS 仿真、综合或板测通过。
