# 数据、checkpoint 与硬件交接

所有路径相对本仓库根目录。数据和生成物不随源码克隆提供，脚本不会假设已有个人目录或其他工程。

## 输入与 checkpoint

| 模型 | 数据位置 | QAT 初始化 |
|---|---|---|
| MNIST | `CNN_FPGA/data/` | 默认 `CNN_FPGA/checkpoints/float_model.pth`，可用训练参数覆盖 |
| YOLO | `YOLOV2_TINY/data/VOCdevkit/` | 用 `--float-checkpoint` 选择；兼容默认名为 `official_voc_fp32.pth` |

MNIST 加载器支持下载；YOLO 读取预先准备的 VOC2007 trainval/test 和 VOC2012 trainval。checkpoint 由 PyTorch 保存，MNIST 包含 `model_state_dict`、epoch、accuracy；YOLO 还包含模型构造 `config` 和 mAP，QAT 包含明确方案标记。原始 Darknet `.weights`、其他框架文件不能直接当成本工程 checkpoint。

只加载可信来源的 checkpoint。哈希用于标识文件，不构成真实性或安全性保证。旧模型迁移需明确说明，不能用 `strict=False` 掩盖量化边界不匹配。

## 导出产物

| 模型 | 产物 | 规范 |
|---|---|---|
| MNIST | NPY、metadata v5、布局 v2 的 header、逐图 trace header | [MNIST 规范](mnist.md) |
| YOLO | NPY、metadata v2、权重布局 v3、27 个模型 BIN、每图 43 个 trace BIN | [YOLO 规范](yolov2_tiny.md) |

参数与 trace 必须来自同一个 checkpoint。更换模型后重新生成整套数据，不混用输入 scale、anchors、权重、bias/M/S 和输出参考。导出的 checkpoint 路径只作为相对来源标识，真正校验标识为 SHA256；读取方不应依赖训练时的文件位置。

## 与加速器仓库对接

软件与硬件是独立仓库，不要求特定相邻目录名。将 YOLO 同一次导出的以下内容放入加速器仓库约定位置：

| 软件产物 | 加速器仓库中的目标位置 |
|---|---|
| `YOLOV2_TINY/generated/bins/` 中的 27 个文件 | `data/yolov2_tiny/bins/` |
| `YOLOV2_TINY/generated/trace/` 中的配套 trace | `data/yolov2_tiny/trace/` |
| `YOLOV2_TINY/generated/metadata.json` | `data/yolov2_tiny/metadata.json` |

复制目录内容，避免再嵌套一层 `generated/` 或 `bins/`。硬件的 SD 卡、DDR 写回步长和 PS 配置生成流程以其自身文档为准：软件 trace 始终保存逻辑 HWC，Prediction 是紧凑 `13×13×125`，不能假定与硬件补齐后的存储步长相同。

这里保持文件名和 ABI，不在软件工程内写入硬件工程路径或调用硬件工具。模块级比较的数据需求见 YOLO 规范，不需要将训练数据集交给硬件仿真。
