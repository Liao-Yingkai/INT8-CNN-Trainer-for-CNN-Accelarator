# YOLOv2-tiny VOC PyTorch 工程

详细网络、量化和导出规范；快速开始见 [YOLO 说明](../YOLOV2_TINY/README.md)，仓库导航见 [根说明](../README.md)。文中命令除特别说明外在 `YOLOV2_TINY/` 目录执行。

当前提供 FP32 训练、官方 eager-mode QAT 微调、VOC2007 mAP50 评估，以及供 HLS 使用的 INT8/INT32 NPY/BIN 参数导出。脚本职责、训练模式和 checkpoint 命名与 `MNIST/` 保持一致；本目录的导出格式独立于 MNIST。

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

依赖见根目录 `requirements.txt`。将 VOC2007 trainval/test、VOC2012 trainval 解压到：

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

程序读取本地数据，不自动下载数据集。`config.yaml` 的 `data.root` 可指向已有 VOCdevkit。默认训练使用 2007 train + 2012 train；验证使用两年的 val；最终评估使用 2007 test。官方预训练权重的训练数据通常包含这些 val，所以它们对当前官方权重初始化不属于严格独立的验证集；训练日志的验证 mAP 不能与 test mAP 混用，也不应利用 test 反复调参或选择 checkpoint。

在 `YOLOV2_TINY/` 中手动执行：

```text
python train.py --mode float
python train.py --mode qat --float-checkpoint checkpoints/float_model_best.pth
python evaluate.py --float-checkpoint checkpoints/float_model_best.pth --qat-checkpoint checkpoints/qat_model_best.pth
```

训练和评估可用 `--device` 选择设备、`--config` 选择配置。QAT 用 `--float-checkpoint` 指定初始化文件；兼容默认值仍是 `checkpoints/official_voc_fp32.pth`，自行训练时可改选 `checkpoints/float_model_best.pth`。脚本不会自动下载或切换权重；训练会写入固定 checkpoint 名称，运行前请保留需要的旧结果。

`config.yaml` 的 `training` 使用 `float_epochs`、`float_learning_rate`、`qat_epochs`、`qat_learning_rate` 等字段，与 MNIST 的含义一致。默认 QAT 20 个 epoch、学习率 0.0001；从零开始计数的 epoch 15 起冻结 Observer。

`quant.py` 提供 `create_fpga_qconfig()`、`prepare_qat_model()`、`load_qat_model()` 和 `save_checkpoint()`，与 MNIST 的接口职责相同。采用官方 FakeQuant：activation 为 per-tensor symmetric signed INT8，weight 为 axis=0 的 per-channel symmetric signed INT8，zero_point 均为 0。

YOLO 先在 eval 状态把预训练 BN 折叠进 Conv，再执行 `prepare_qat()`；因此 FP32 checkpoint 无需预先融合，QAT 阶段不再更新 BN。新边界详见下文。解码与 NMS 使用浮点运算，QAT 使用官方 FakeQuant 模拟，不会在训练时替换为真实整数算子。

`evaluate.py` 默认在同一测试集分别评估最后一轮 FP32/QAT checkpoint，并打印 mAP 差值；只评估一种模式可使用 `python evaluate.py --mode float` 或 `--mode qat`。评估时关闭 QAT Observer 并保留 FakeQuant，避免测试数据更新 scale。旧 FP32 checkpoint 的 `model_state_dict` 和 `config.model` 可以直接用于 QAT。

默认 FP32 文件仍是自训练 `float_model.pth`。要比较所选 FP32 初始化与新 QAT 的最佳结果，明确指定同源文件（新 QAT 完成后再运行）：

```text
python evaluate.py --float-checkpoint checkpoints/official_voc_fp32.pth \
  --qat-checkpoint checkpoints/qat_model_best.pth
```

该入口统一使用配置中的 evaluation split（默认 VOC2007 test 全部 4952 张）、相同预处理和 NMS。不要把当前方案结果与旧双边界 QAT 的结果混为一谈。

`dataset.py` 将图像等比例 letterbox 到配置尺寸，框坐标同步缩放和平移，RGB 像素缩放到 `[0,1]`；训练增加随机水平翻转、亮度和颜色增强。标签内部使用归一化 `xyxy + class_id`，保留 difficult 标记。

`loss.py` 实现 YOLOv2 风格损失：按宽高 IoU 选择 anchor，回归 sigmoid 中心偏移和 log 宽高，采用 IoU objectness 目标、重叠负样本忽略以及类别交叉熵。该实现不声称逐项复现 Darknet 的训练数值或 warmup；同一 cell/anchor 冲突时后一个目标覆盖前一个。difficult 对象不参与训练损失。

`evaluate.py` 将 raw logits 解码后执行按类别 NMS（IoU 0.45），按 VOC2007 的 11 点 AP、匹配 IoU 0.5 输出各类 AP 和 mAP。difficult 匹配不计 TP/FP。每图最多保留 300 个检测结果，分数阈值为 0.01。

## 新 QAT 边界与 checkpoint 兼容性

方案标记：`qat_scheme="single_postprocess_v1"`。

```text
QuantStub activation FakeQuant
  → Conv1～Conv8：Conv(weight FakeQuant，BN 已折叠)
                  → LeakyReLU(alpha=0.1) → activation FakeQuant
                  → Pool / Bypass（不新增量化边界）
  → Prediction：Conv(weight FakeQuant) → activation FakeQuant
  → DeQuantStub → raw logits → 原有解码/NMS
```

实际共有 9 个 weight FakeQuant 和 10 个 activation FakeQuant：输入 1 个、LeakyReLU 后 8 个、Prediction 输出 1 个。前八个 Conv 的 `activation_post_process` 是无运算的 `nn.Identity`，不是 Observer/FakeQuant；其 weight-only QConfig 的 weight 仍使用官方 per-channel FakeQuant。这与PyTorch 官方 `default_weight_only_qconfig` 的 Identity 用法一致，但保留本工程的 per-channel 权重配置。所有容器、BN Identity、Pool 默认 qconfig=None，只有明确列出的叶子启用量化。`validate_qat_structure()` 检查真实模块集合；测试还检查 forward 的真实调用。

activation 为 per-tensor symmetric qint8，weight 为原始 OIHW 的 OC 轴（axis=0）per-channel symmetric qint8；范围均为 `[-128,127]`，zero_point 全部为 0。Pool 输出直接沿用输入 scale；Pool6 右/下补边在整数硬件中使用 -128，而非 0。

旧结构在 LeakyReLU 之前和之后各量化一次，中间舍入和饱和不可逆。新结构不是把旧两组 scale 代数合并，**必须从 FP32 重新 QAT 微调**，不保证旧/新模型逐位或精度等价：

```text
# 运行前自行保留需要的旧 checkpoint。
python train.py --mode qat
```

`save_checkpoint()` 仅对通过结构检查的 QAT 写入新方案标记，普通 FP32 不写标记。`load_qat_model()` 先检查标记，再恢复融合后的 QAT 结构并 strict=True 加载；缺少标记、旧标记、额外旧边界参数均不能通过，不使用 strict=False。历史双边界 QAT 不自动迁移，加载时会提示“请从 FP32 重新进行 QAT 微调”。

QAT checkpoint 必须带 `single_postprocess_v1` 标记并通过严格结构恢复。仓库不提供已训练文件；实际兼容性以加载检查为准，不以文件名或训练轮次判断。

## 参数导出命令与范围

```text
python export.py
# 建议部署时明确选新方案的最佳 checkpoint：
python export.py --checkpoint checkpoints/qat_model_best.pth --output-dir generated
```

默认读取本工程 `checkpoints/qat_model.pth`。仅支持 416×416 RGB、VOC 固定 20 类/类别顺序、5 个既定 VOC anchors 和 125 个逻辑输出通道，不宣称支持任意检测网络。历史旧双边界 checkpoint **无法正式导出**，新方案文件通过严格恢复和参数检查后才可导出。预处理仍为 RGB、等比例 letterbox、像素 `[0,1]`，不使用 MNIST mean/std。

导出流程：通过更新后的 `load_qat_model()` 严格恢复 → eval、禁用所有 Observer → 用已初始化的官方 Observer `calculate_qparams()` 复核 qparams → 保留 checkpoint 实际使用的 FakeQuant scale/zero_point → 校验全部 9 层并准备数组 → 保存 NPY/BIN → 最后写 metadata。GPU 保存的 scale 与 CPU 重算可能相差一个 float32 ULP，只允许相邻一个可表示值的误差；更大差异仍拒绝，zero_point 必须严格一致。不会用 CPU 重算值覆盖已训练 scale，也不会放宽权重整数逐元素一致的检查；metadata 记录 scale 来源和复核规则。Exporter 不对已经恢复的卷积重复折叠 BN，不喂新图像更新 scale，也不调用 `torch.ao.quantization.convert()`。

空/未初始化 Observer、无效 scale、非零 zero_point、错误 qscheme/轴/范围、关闭的 FakeQuant、权重逐元素核对失败、INT32 bias/M/S 溢出或负 shift 均明确报错。层参数错误包含层名，且在全部验证通过前不会创建输出文件。

### 输出文件与类型

每个 `<layer>` 为 `conv1`～`conv8` 或 `prediction`：

```text
generated/
├── weights/<layer>_weight_int8.npy       # 逻辑 [Kh*Kw*Cin,Cout]，INT8
├── biases/<layer>_bias_int32.npy         # [Cout]，INT32
├── quant_params/<layer>_multiplier_int32.npy  # [Cout]，INT32
├── quant_params/<layer>_shift_int32.npy       # [Cout]，INT32 总右移量
├── quant_params/<layer>_weight_scale.npy     # [Cout]，float64
├── quant_params/<layer>_weight_zero_point.npy # [Cout]，INT32，全 0
├── quant_params/<layer>_input_scale.npy      # [1]，float64
├── quant_params/<layer>_input_zero_point.npy # [1]，INT32，全 0
├── quant_params/<layer>_output_scale.npy     # [1]，float64
├── quant_params/<layer>_output_zero_point.npy # [1]，INT32，全 0
├── bins/<layer>_weight.bin              # 物理 Pack，无头 INT8 原始字节
├── bins/<layer>_bias.bin                # [Cout]，little-endian INT32
├── bins/<layer>_quant.bin               # M完整数组 → S完整数组，little-endian INT32
└── metadata.json
```

scale 从 PyTorch qparams 读取后以 float64 保存，不会增加原始 qparams 的有效精度；它们用于离线分析和核对，不供 FPGA 浮点计算。bias_scale 和实数 r 保存在 metadata，不再导出第二组中间 scale、正负分支参数或 `M_negative` 文件。不生成巨型 `.h`、DDR 内存分配表、DMA 或 Feature Map 物理数据。

文件分类和命名对齐 MNIST：逻辑 NPY 分别放在 `weights/`、`biases/`、`quant_params/`；硬件数据集中存放，MNIST 的集中目录是 `headers/`，YOLO 使用 `bins/`。二进制文件使用相同的 `<layer>_weight`、`<layer>_bias`、`<layer>_quant` 命名。所有 `.bin` 只放在 `bins/`，bias 独立，multiplier/shift 合并，不再生成 `params/` 或 `*_weight_packed.bin`。

### 权重和 bias 的数值来源

权重来自已折叠 BN 的 QAT Conv FP32 参数，不是直接将 FP32 文件改成 INT8 类型。以每个 OC 的 weight scale 执行：

```text
weight_int8[oc] = clamp(round_nearest_even(weight_float[oc]/weight_scale[oc]), -128, 127)
bias_scale[oc] = input_scale * weight_scale[oc]
bias_int32[oc] = round_nearest_even(bias_float[oc]/bias_scale[oc])
```

权重运算保持PyTorch CPU FakeQuant 内核的 FP32 reciprocal 乘法顺序（数学上是除以 scale），避免 float64 除法在半整数附近改变整数。导出后与实际 weight FakeQuant 所表示的整数逐元素核对，不接受仅网络精度接近。bias 用 float64 计算，转换前核查 INT32 范围；不存在 bias 时为 Cout 个零，BN 折叠产生的 bias 则必须保留。

输入 scale：Conv1 取 QuantStub；Conv2～Conv8 取前一块 LeakyReLU 后的 FakeQuant；Prediction 取 Conv8 LeakyReLU 后的 FakeQuant。输出 scale：Conv1～Conv8 取自身 LeakyReLU 后的 FakeQuant；Prediction 取预测卷积输出的 FakeQuant。Pool 不改变 scale。绝不使用已删除的激活前 scale，也不用手工 max_abs/127 替代官方 qparams。

### 一次性 PostProcess 的 M/S 和整数契约

每个输出通道仅有一份 bias、M、S：

```python
r = input_scale * weight_scale / output_scale
significand, exponent = frexp(r)
M = round_nearest_even(significand * 2**31)
if M == 2**31:
    M //= 2
    exponent += 1
S = 31 - exponent
# r ≈ M / 2**S；S 是总右移量，不再额外右移 31 位。
```

M 和 S 都是 signed INT32，M 非负，S 必须非负。遇到负 shift 或溢出报错，不能裁剪参数或改为其他算法。`alpha=0.1` 不进入 M 或 DDR 参数数组，固定常量及派生规则为：

```text
ALPHA_MULTIPLIER = 1717986918
ALPHA_SHIFT = 34
M_positive = M
M_negative = (INT64(M)*1717986918 + 2^33) >> 34
```

负系数派生使用最近值 half-up，不重新归一化，正负分支共用 S。每个有效输出只执行一次：

```text
a = accumulator + bias                         # 相加前后均需满足 INT32 契约
M_selected = leaky_relu && a<0 ? M_negative : M_positive
product = INT64(a) * INT64(M_selected)
scaled = product arithmetic-right-shift S       # 不加 rounding offset
output = clamp(scaled, -128, 127)               # 最终才得到 INT8
```

负数算术右移向负无穷取整：`-9 >> 1 = -5`，不是 C/C++ 整数除法向零截断的 -4，也不是最近偶数舍入。禁止先将中间量转为 INT8、先对负 INT8 除以 10，或分两次量化。Prediction 无 LeakyReLU，正负都用 M_positive，不添加 ReLU 或十分类 Softmax。

`export.py` 的 `integer_postprocess()` 是这一小段数值契约的 Python 整数参考，并非完整 YOLO 整数 Golden Model。QAT 仍保留 PyTorch 的最近偶数舍入；硬件最终右移直接截断，两者还存在 FP32/INT32 bias 和系数近似差异，**不能宣称 QAT 与整数推理天然逐位一致**。

### 权重逻辑布局和物理布局

逻辑 NPY：原始 OIHW 经 `transpose(2,3,1,0).reshape(Kh*Kw*Cin,Cout)`，顺序 `kh → kw → cin → cout`。

物理 BIN：`G=ceil(Cin/8)`、`R=Kh*Kw*G`，shape 为 `[ceil(Cout/4), R, 4, 8]`，展开顺序为：

```text
global_cout_tile → kh → kw → cin_group → cout_lane → cin_lane
q = token // G
g = token % G
ic = g*8 + cin_lane
oc = global_cout_tile*4 + cout_lane
packed[tile,token,cout_lane,cin_lane] = logical[q*Cin+ic,oc]  # ic<Cin 且 oc<Cout
# 其余元素严格为 0。
```

每个 kernel 位置单独补齐 Cin，不能跨 kernel 位置拼接余下通道。Conv1 每个位置的 3 个 Cin 补到 8；Prediction 的 125 个 Cout 在权重中补到 128，但网络检测头仍是 125。暂定的 32-Cout 缓存只是最多 8 个连续 Tile 的运行时读取范围；文件没有 cache_block 维，也不为填满缓存额外导出 Tile。

| 层 | 原始 OIHW | 逻辑 [K,Cout] | Packed Shape |
|---|---|---|---|
| conv1 | [16,3,3,3] | [27,16] | [4,9,4,8] |
| conv2 | [32,16,3,3] | [144,32] | [8,18,4,8] |
| conv3 | [64,32,3,3] | [288,64] | [16,36,4,8] |
| conv4 | [128,64,3,3] | [576,128] | [32,72,4,8] |
| conv5 | [256,128,3,3] | [1152,256] | [64,144,4,8] |
| conv6 | [512,256,3,3] | [2304,512] | [128,288,4,8] |
| conv7 | [1024,512,3,3] | [4608,1024] | [256,576,4,8] |
| conv8 | [1024,1024,3,3] | [9216,1024] | [256,1152,4,8] |
| prediction | [125,1024,1,1] | [1024,125] | [32,128,4,8] |

### BIN 位序、偏移与参数读取

权重文件是 INT8 补码原始字节，NumPy C 连续顺序，无 NPY 头。每 Pack 严格 32 字节：byte0～7 是 Cout Lane0 的 8 个 Cin，byte8～15 是 Lane1，依次类推。每 Lane 的 Cin0 地址最低；作为 64-bit word 读取时，较早字节在较低 bit（little-endian）。不能直接序列化 Python/C++ 结构体内存。

```text
byte_offset = 32*(global_cout_tile*R + token) + 8*cout_lane + cin_lane
weight_bin_bytes = ceil(Cout/4)*R*32
```

上式是**字节偏移**，不是 64-bit word 偏移；需要 word 索引时必须另外除以 8。Exporter 不生成任何 DDR 物理地址。

bias 和 quant BIN 都为明确的 little-endian signed INT32（NumPy `<i4`）：

```text
bins/<layer>_bias.bin：bias[0:Cout]
  bias byte_offset=0，文件长度=Cout*4

bins/<layer>_quant.bin：multiplier[0:Cout] → shift[0:Cout]
  multiplier byte_offset=0，shift byte_offset=4*Cout，文件长度=2*Cout*4
```

quant 使用 field-first，不是逐通道交错 M/S；所有偏移相对于各自文件起点。参数不按物理 Cout 补齐。Prediction 各字段只有 125 个元素，bias BIN 为 500 字节，quant BIN 为 1000 字节；不保存 M_negative、alpha 或无效 Cout 参数。**读取方必须先判断 oc<Cout，再读 bias/M/S，不能越界读取后才屏蔽 cout_valid。** 所有文件大小由实际数组 nbytes 生成并核验；NPY 的文件字节数另含头部，不等于其 data_nbytes。

### Metadata

`metadata.json` 使用独立的 `format_version=2`、`weight_layout_version=3`，不是 MNIST 的版本号。v2 只改变文件分类和参数拆分，不改变 QAT 方案、权重物理布局或数值算法，因此不要求重新训练。v1 的 `weights/*_weight_packed.bin` 和 `params/*_params.bin` 不再输出；读取方必须改用 metadata 中的 `files.weight_bin`、`files.bias_bin`、`files.quant_bin` 描述。特别是 quant 文件的 M 从偏移 0 开始，不能再使用旧 params 文件的偏移。已有 v1 输出目录会明确拒绝覆盖，请另选空目录或先备份整理，避免新旧文件混用；导出器不会擅自删除旧文件。包含：

- qat_scheme、checkpoint 路径/SHA256/epoch、固定网络配置、层顺序。
- OIHW/逻辑/物理 shape，Cin/Cout/G/R、逻辑和物理通道数、卷积 geometry。
- 文件相对路径、dtype、C 顺序、字节序、data_nbytes 和实际 file_size_bytes；`bias_bin` / `quant_bin` 各自的 fields、count_per_field 和 field_byte_offsets。
- weight 的原始 per-channel 轴、scale/zero_point；input/output scale/zero_point、bias_scale、实数 r、实际 M/S。
- 固定 alpha 常量、负系数派生公式及其 half-up 舍入、最终算术右移截断语义、各层是否启用 LeakyReLU。
- PyTorch 逻辑 Feature 为 NCHW，硬件 stream 顺序为 `H→W→group8→lane8`；`feature_ddr_layout=not_frozen`。软件导出不规定完整 Feature DDR 布局。

### 函数职责与轻量测试

`extract_fake_quant_params()` 验证并读取官方 qparams；`quantize_weight_per_channel()` 量化并与实际 FQ 核对；`convert_bias_to_int32()` 检查 bias 缩放和溢出；`calculate_requant_params()` 生成 Q31 M/S；`pack_weight()` 做带 kernel 内补零的物理打包；`collect_layers()` 在写入前检查全部参数；`write_layer()` 保存文件及描述；`export_checkpoint()` 负责严格加载、冻结和完整导出。整数参考由 `integer_postprocess()` 提供。

```text
python ../scripts/run_tests.py --model yolov2_tiny
python -m py_compile model.py quant.py train.py evaluate.py export.py
git diff --check
```

测试只在临时目录创建随机合成 FP32/QAT checkpoint、随机输入校准和导出文件，覆盖 416 forward 的 19 个 FQ、FP32 初始化、严格恢复及旧结构拒绝、九层 Shape/BIN 全量回读、负补码、G=1/2/128、Cin/Cout 补零、little-endian field-first 参数、整数正负分支/截断/饱和及错误参数拒绝。随机校准不是训练，测试通过不代表检测精度达标。

checkpoint、模型参数和 trace 不随源码分发。结构与 BIN 测试只验证对应软件契约，不构成精度或硬件验证结果。

## 导出 FPGA 模块边界 Trace

`dump_trace.py` 与 MNIST 的用途相同：从测试集选图，输出用于 Im2Row、Systolic、PostProcess、Pool 的功能比较数据。但 YOLO **只导出 BIN**，不生成 `.h`、NPY 或新的 JSON。它直接读取现有 `metadata.json` 和 `bins/` 的权重、bias、M/S，不读取参数 NPY、不恢复 FP32/QAT checkpoint，也不会改变 Observer、scale 或参数文件。

在 `YOLOV2_TINY/` 中运行：

```text
# 先准备与所选 checkpoint 对应的参数；已有最新参数时不必重复运行。
python export.py --checkpoint checkpoints/qat_model_best.pth

# 默认从本地 VOC2007 test 无放回抽取 1 张，seed=1。
python dump_trace.py

# 抽取 3 张，文件按导出顺序添加 _0000、_0001、_0002 后缀。
# 重复运行请使用不同的 trace 输出目录，避免覆盖已有文件。
python dump_trace.py --num-samples 3 --seed 1 --output-dir generated/trace_batch3

# 验证另一个独立参数目录；默认 trace 路径跟随 generated-dir。
python dump_trace.py --generated-dir generated/experiment --output-dir generated/experiment/trace
```

`-n`、`--num-samples`、`--num-images` 等价，默认数量 1；`--seed` 默认 1。数据路径沿用 `config.yaml` 的 `data.root`，固定使用 VOC2007 test、416×416、无增强，不下载数据。RGB/letterbox/像素 `[0,1]` 预处理复用 `dataset.py`，随后使用导出的输入 scale、FP32 reciprocal、最近偶数舍入和 `[-128,127]` 饱和得到输入 INT8；不使用 MNIST mean/std。

后缀表示导出顺序，不是数据集索引；程序在终端打印 seed、抽样索引及 VOC 图像 ID，没有额外标签/metadata 文件。相同数量和 seed 重现同一顺序。样本数必须在 1～4952 内；指定多个样本时不重复抽取。需要保留索引记录时可自行保存终端日志。

### 文件分类和模块边界

所有 trace BIN 平铺在 `generated/trace/`；参数仍在 `generated/bins/`，不混入 trace。每张图片固定 **43 个文件**：输入 1 个、9 个卷积阶段各 4 个、Pool1～Pool6 各 1 个。层名为 `conv1`～`conv8`、`prediction`。

```text
generated/trace/
├── trace_input_int8_hwc_0000.bin
├── trace_conv1_unfold_int8_0000.bin
├── trace_conv1_im2row_int8_0000.bin
├── trace_conv1_systolic_int32_hwc_0000.bin
├── trace_conv1_postprocess_int8_hwc_0000.bin
├── trace_pool1_int8_hwc_0000.bin
├── ...conv2～conv8（Pool 仅到 pool6）
├── trace_prediction_unfold_int8_0000.bin
├── trace_prediction_im2row_int8_0000.bin
├── trace_prediction_systolic_int32_hwc_0000.bin
└── trace_prediction_postprocess_int8_hwc_0000.bin
```

| 文件内容 | 类型 | shape / 含义 |
|---|---|---|
| input | INT8 | `[416,416,3]`，逻辑 HWC |
| unfold | INT8 | `[H*W,Kh*Kw*Cin]`，逻辑输入 patch 矩阵 |
| im2row | INT8 | `[H,T,R,4,8]`，硬件 Activation Pack |
| systolic | INT32 | `[H,W,Cout]`，完整 Cin 归约后的 MAC，不含 bias |
| postprocess | INT8 | `[H,W,Cout]`，bias + LeakyReLU 分支 + requant + 饱和的最终结果 |
| pool | INT8 | `[Hout,Wout,Cout]`，Pool 的最终结果 |

**bias、LeakyReLU、requant 都属于一次性 PostProcess，不单独导出 bias_add、INT32 激活结果或独立 requant 文件。** Prediction 只做 bias 和 requant，不启用 LeakyReLU，也不执行检测解码、NMS、ReLU 或 Softmax；最后一个文件是 125 通道的量化 raw logits。

### 完整尺寸表

这里 H/W 指卷积输出空间尺寸（这些卷积 stride=1，等于本层输入尺寸），不是后续 Pool 的尺寸。`G=ceil(Cin/8)`、`T=ceil(W/4)`、`R=Kh*Kw*G`。

| 阶段 | H=W | Cin → Cout | unfold shape | im2row shape | 后续 Pool H=W |
|---|---:|---|---|---|---:|
| conv1 | 416 | 3 → 16 | [173056,27] | [416,104,9,4,8] | 208 |
| conv2 | 208 | 16 → 32 | [43264,144] | [208,52,18,4,8] | 104 |
| conv3 | 104 | 32 → 64 | [10816,288] | [104,26,36,4,8] | 52 |
| conv4 | 52 | 64 → 128 | [2704,576] | [52,13,72,4,8] | 26 |
| conv5 | 26 | 128 → 256 | [676,1152] | [26,7,144,4,8] | 13 |
| conv6 | 13 | 256 → 512 | [169,2304] | [13,4,288,4,8] | 13 |
| conv7 | 13 | 512 → 1024 | [169,4608] | [13,4,576,4,8] | 无 |
| conv8 | 13 | 1024 → 1024 | [169,9216] | [13,4,1152,4,8] | 无 |
| prediction | 13 | 1024 → 125 | [169,1024] | [13,4,128,4,8] | 无 |

### 数据顺序、补零和字节偏移

所有文件无文件头，采用 C 连续顺序；INT8 为原始 signed 补码字节，INT32 明确为 little-endian `<i4`。程序打印每个文件的 shape、dtype 和实际字节数，不序列化 C++ 结构体、Valid 字段或对齐填充。

- **HWC：**元素索引 `((y*W+x)*C+c)`；INT8 字节偏移等于元素索引，Systolic INT32 字节偏移再乘 4。HWC 文件只包含真实空间位置和逻辑 Cout，不补 Prediction 的 3 个无效 Cout。
- **unfold：**行号为 `y*W+x`，列号为 `((kh*Kw+kw)*Cin+ic)`；与逻辑权重 `[Kh*Kw*Cin,Cout]` 的 K 顺序一致，没有 Cin Group 补齐。
- **im2row：**轴顺序为 `y→tile_x→kh→kw→cin_group→spatial_lane→channel_lane`，其中 `token=(kh*Kw+kw)*G+group`。每个 Pack 的 4 个 Lane 是同一行的 4 个空间位置，每个空间 Lane 内为 8 个连续通道，不是 4 行图像。

```text
im2row byte_offset = 32*((y*T+tile_x)*R+token) + 8*spatial_lane + channel_lane
output_x = tile_x*4 + spatial_lane
source_y = y + kh - padding
source_x = output_x + kw - padding
source_c = group*8 + channel_lane
```

3×3 的 padding=1，1×1 的 padding=0。空间输出位置不存在、源坐标处于卷积 Padding 或 Cin 通道无效时，im2row 数据为 0。每个 kernel 位置独立补齐 Cin，不能跨 kernel 拼接剩余通道；空间 Tile 不跨图像行。W=13 的每行最后一个 Tile 仅 spatial Lane0 有效；W=26 的末 Tile 仅前两个 Lane 有效。

im2row 每个空间 Tile 只导出一次，Pack 数为 `H*T*R`、文件字节数为 `H*T*R*32`；不为每个 Cout Tile 或缓存块重复写入相同的激活数据。先写入文件的字节对应 Pack 较低 bit；每个空间 Lane 的 8 个 Cin 可按 little-endian 64-bit word 读取。

### 数值参考与 HLS 对比方式

权重从物理 BIN 解包为逻辑矩阵；用标准 PyTorch unfold 重排 patch，通过矩阵乘法求 MAC，不模拟 PE、错拍、Replay、FIFO 或逐周期时序。为避免通用 INT64 GEMM 的速度问题，CPU 上用 float64 GEMM 求和：INT8 操作数的所有乘积和任意部分和都有 `K*128*128` 的绝对界，本网络最大 K=9216，远小于 `2^53`，因此可精确表示整数结果。输出仍须通过整数性和 INT32 范围检查；测试用独立 INT64 MAC 做逐元素核对，不使用 float32/TF32 近似 MAC。

PostProcess 直接复用 `export.py` 的 `integer_postprocess()`：检查 `accumulator+bias` 能装入 INT32，按固定 alpha 派生负系数，INT64 乘法后算术右移，不加 rounding offset，最终饱和为 INT8。Pool1～Pool5 为 2×2/stride2、无补边；Pool6 为 2×2/stride1，仅右侧和下侧补 -128。后续层始终消费前一层的整数参考结果，不用 QAT 输出替换中间结果。

HLS TB 比较方法：

1. Im2Row 输入由当前层逻辑 HWC Feature 按 `y→x→group→lane` 组装 8 通道 Pack；Cin 尾部填零，不将卷积空间 Padding 作为输入像素。
2. Im2Row 输出按顺序读取 `im2row` BIN，每个事务比较 32 字节。spatial_valid 由 `tile_x*4+s<W` 推导；卷积 Padding 不使有效输出位置失效。
3. Systolic/PostProcess/Pool 输出按自身事务中的空间坐标及全局 Cout 索引访问对应 HWC 文件；先判断空间和 Cout 有效性，再读取参考数组。对无效 Lane 的零 payload 和 mask 单独按硬件契约检查，不能越界读 HWC 文件。
4. Cout 缓存块、Replay 和重复扫描由 TB/Top 调度解释，不改变这些数学参考文件；不要将逻辑 HWC 当作冻结后的完整 Feature DDR 物理布局。

已有同名目标文件时，在处理图片前拒绝写入；不会清空旧 trace 或覆盖参数。运行中途若因磁盘/数值错误失败，之前已写的 BIN 可能保留，应另选输出目录后重试。

Trace 绑定生成时的整套参数。重新导出不同 checkpoint 后，旧 trace 不能与新 `bins/` 混用；应保留对应参数目录并使用新的 trace 输出目录重新生成。建议利用 `export.py --output-dir` 和 `dump_trace.py --generated-dir` 保存独立实验，不覆盖正在用于 HLS 比较的参数。

### Trace 测试和验证范围

```text
python ../scripts/run_tests.py --model yolov2_tiny
python -m py_compile dump_trace.py
git diff --check
```

`tests/yolov2_tiny/test_trace.py` 使用临时合成 BIN，验证 G=1/2/128、3×3/1×1、unfold 和 im2row 的独立坐标参考、RGB/Cout/空间尾部、长归约 MAC 的 INT64 对照、负值 Pool 边界、输入舍入、采样、43 个模块边界、BIN 回读、字节序、已有文件保护及错误参数拒绝；不运行训练。

真实样本 smoke 需要自行准备 VOC 和配套模型参数，按上述命令生成后检查所有文件。它不代表 mAP；QAT 与整数参考可能因 bias 量化、系数近似和舍入规则不同产生误差。硬件仿真和精度评估应单独报告。
