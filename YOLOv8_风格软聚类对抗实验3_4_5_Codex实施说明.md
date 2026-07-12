# YOLOv8 风格软聚类对抗实验（实验 3/4/5）组件化实现说明

> 目标项目：`https://github.com/HH276/yolov8-pytorch-master/releases/tag/v1.0`  
> 参考训练入口：`train_5datasets.py`  
> 参考项目内已有设计：`三组件详解.txt`、`dg_config.py`  
> 使用对象：Codex / 代码实现人员  
> 文档目的：在现有“风格硬聚类 + 三个风格单域专家 + 三个判别器”实验基础上，实现实验 3、实验 4、实验 5，并允许后续仅通过修改 `dg_config.py` 切换实验与参数。

---

## 1. 实施前必须先检查的项目内容

在修改代码之前，先完整阅读并理解以下文件，不要直接覆盖已有逻辑：

1. `三组件详解.txt`
2. `dg_config.py`
3. `train_5datasets.py`
4. 当前对抗组件相关代码
5. 当前三个单域专家模型的加载代码
6. 当前三个判别器的定义、损失计算和优化器更新代码
7. 当前数据集类与 `collate_fn`
8. 当前硬风格标签的读取方式
9. 当前日志目录、权重保存和验证流程

实现时应尽量沿用项目已有的：

- 命名风格；
- 配置读取方式；
- 组件开关方式；
- 模型加载方式；
- 训练日志方式；
- 多卡训练方式；
- 冻结与解冻训练方式；
- `train_5datasets.py` 中的数据加载和训练循环。

本任务不要求重写整个项目，也不要求替换原有硬聚类实验。应以**最小侵入方式**增加软风格路由组件，并确保原有实验仍能复现。

---

# 2. 当前已有基础

当前已完成风格硬聚类实验，基本流程如下：

1. 不读取 `251030_train.txt`。
2. 分别读取：
   - `India_train.txt`
   - `Japan_train.txt`
   - `USA_train.txt`
3. 使用 `yolov8_s.pth` 提取每张图片的 `feat1`，输入尺寸为 640 时对应约 `80×80`。
4. 对 `feat1` 计算通道均值和标准差：
   \[
   s_i=[\mu_i,\sigma_i]
   \]
5. 合并三个国家的训练风格向量。
6. 在合并后的训练风格向量上拟合 `StandardScaler`。
7. 在标准化后的训练特征上拟合 K-means++，设置 `K=3`。
8. 根据最近聚类中心得到硬风格标签：
   - `Style_0`
   - `Style_1`
   - `Style_2`
9. 生成三个风格域训练集和验证集。
10. 已训练得到三个风格单域专家权重。
11. 在后续对抗阶段，使用三个风格专家和三个判别器。

当前已有的三个风格专家应继续复用，例如：

```text
style_0_best.pth
style_1_best.pth
style_2_best.pth
```

实际文件路径以项目现有配置为准，不得在代码中写死。

---

# 3. 本次任务目标

在不改变 `K=3`、不改变已有三个风格专家定义的前提下，实现以下四种运行模式：

| 模式 | 含义 | 每张图参与分支数 | 聚类置信度 |
|---|---|---:|---|
| `hard` | 原风格硬聚类对抗基线 | 1 | 否 |
| `soft_dense` | 实验 3：完整软风格权重 | 3 | 否 |
| `soft_top2` | 实验 4：Top-2 稀疏风格路由 | 2 | 否 |
| `soft_top2_conf` | 实验 5：Top-2 + 聚类置信度 | 2 | 是 |

实现完成后，正式消融实验阶段应做到：

```text
只修改 dg_config.py
        ↓
运行同一个训练入口
        ↓
自动切换 hard / soft_dense / soft_top2 / soft_top2_conf
```

除首次生成聚类距离缓存外，不应要求用户为每个实验重新修改训练代码或重新提取特征。

---

# 4. 非目标

本任务暂不实现以下内容：

1. 不实现 EMA Teacher。
2. 不在线更新聚类中心。
3. 不在线重新计算风格标签。
4. 不把 K-means 替换为 GMM、模糊 C 均值或可学习聚类。
5. 不改变风格域数量 `K=3`。
6. 不重新训练三个风格专家，前提是软权重的主标签与原硬标签一致。
7. 不把聚类置信度应用到正常 YOLO 检测损失。
8. 不要求第一次实现就完成真正的稀疏专家子 batch 加速；应先保证算法正确。

---

# 5. 核心定义

## 5.1 风格距离

对于第 \(i\) 张图片，标准化后的风格向量为 \(x_i\)，三个 K-means 中心为：

\[
c_0,c_1,c_2
\]

图片到三个中心的欧氏距离为：

\[
d_{ik}=\|x_i-c_k\|_2
\]

建议缓存平方距离：

\[
r_{ik}=d_{ik}^2
\]

原因是后续软权重直接使用平方距离，同时更换温度参数时不需要重新提取特征。

---

## 5.2 软风格权重

使用训练阶段固定的距离尺度 \(s_d\) 和温度参数 \(\tau\)：

\[
q_{ik}
=
\frac{
\exp\left(-r_{ik}/(\tau s_d+\varepsilon)\right)
}{
\sum_{j=0}^{2}
\exp\left(-r_{ij}/(\tau s_d+\varepsilon)\right)
}
\]

其中建议：

\[
s_d=
\operatorname{median}_i\left(\min_k r_{ik}\right)
\]

注意：

- `distance_scale` 只在训练集上计算一次；
- 验证集和后续训练直接读取训练阶段保存的数值；
- 温度必须大于 0；
- softmax 前应减去每行最大值，防止数值溢出。

---

## 5.3 主风格标签一致性

原 K-means 硬标签：

\[
z_i=\arg\min_k d_{ik}
\]

软权重恢复的主标签：

\[
\hat z_i=\arg\max_k q_{ik}
\]

在使用同一个 scaler、同一个 K-means 和同一组距离时，应满足：

\[
z_i=\hat z_i
\]

程序必须实现一致性检查。

默认要求：

```text
训练集一致率 = 100%
验证集一致率 = 100%
```

如果不一致，立即终止训练并输出不一致图片路径、原硬标签、软标签和距离。

---

# 6. 四种路由模式

## 6.1 `hard`

使用原硬标签构造 one-hot 权重：

\[
w_{ik}=
\begin{cases}
1,&k=z_i\\
0,&k\ne z_i
\end{cases}
\]

每张图片只参与一个风格专家和一个判别器。

---

## 6.2 `soft_dense`：实验 3

使用完整软权重：

\[
w_{ik}=q_{ik}
\]

每张图片参与三个风格分支：

\[
\mathcal L_i^{adv}
=
\sum_{k=0}^{2}
q_{ik}\mathcal L_{ik}^{adv}
\]

---

## 6.3 `soft_top2`：实验 4

对完整概率 \(q_i\) 选择最大的两个分量，其他分量设为 0，再重新归一化。

定义：

\[
m_{ik}=
\mathbf 1\left[k\in Top2(q_i)\right]
\]

\[
\hat q_{ik}
=
\frac{q_{ik}m_{ik}}
{\sum_jq_{ij}m_{ij}+\varepsilon}
\]

最终：

\[
w_{ik}=\hat q_{ik}
\]

每张图片的权重和仍应满足：

\[
\sum_kw_{ik}=1
\]

---

## 6.4 `soft_top2_conf`：实验 5

先使用**完整的三个软概率**计算归一化熵：

\[
H_i=
-\sum_{k=0}^{2}q_{ik}\log(q_{ik}+\varepsilon)
\]

\[
\bar H_i=\frac{H_i}{\log 3}
\]

定义原始聚类置信度：

\[
c_i=1-\bar H_i
\]

设置置信度下限：

\[
\tilde c_i=
c_{min}+(1-c_{min})c_i
\]

可选幂次：

\[
c_i^{final}=\tilde c_i^\gamma
\]

最终分支权重：

\[
w_{ik}=c_i^{final}\hat q_{ik}
\]

必须保证计算顺序为：

```text
完整 q
  ↓
使用完整 q 计算置信度
  ↓
Top-2 截断
  ↓
Top-2 重新归一化
  ↓
乘以样本置信度
```

禁止先 Top-2 再计算熵，否则会人为提高低置信样本的置信度。

---

# 7. 对抗结构说明

设：

- 共享主检测器 / Student：\(G\)
- 三个冻结风格专家：\(E_0,E_1,E_2\)
- 三个判别器：\(D_0,D_1,D_2\)

第 \(k\) 个判别器接收两种特征：

\[
F_i^G=G_f(x_i)
\]

\[
F_{ik}^E=E_k(x_i)
\]

其中：

- 专家特征作为固定参考；
- Student 特征为需要适配的共享特征；
- 专家权重冻结；
- 真正的对抗双方是 \(G\) 与 \(D_k\)；
- \(E_k\) 不是参与更新的对抗方。

必须保持原项目现有判别器标签定义。如果原项目规定：

```text
专家特征标签 = 1
Student 特征标签 = 0
```

则继续沿用，不得在新组件中擅自反转。

---

# 8. 判别器损失

对于第 \(i\) 张图片、第 \(k\) 个分支：

\[
\ell_{ik}^{D}
=
BCE(D_k(F_{ik}^{E}),1)
+
BCE(D_k(\operatorname{detach}(F_i^G)),0)
\]

统一加权损失：

\[
\mathcal L_D
=
\frac{
\sum_i\sum_k
w_{ik}\ell_{ik}^{D}
}{
2\sum_i\sum_kw_{ik}+\varepsilon
}
\]

如果现有项目不是 BCE，而是其他对抗损失，应保留原损失形式，只替换：

- 样本路由；
- 分支加权；
- 全局归一化。

重要要求：

- 更新判别器时必须对 Student 特征 `detach()`；
- 风格专家必须冻结；
- 路由权重必须 `detach()`；
- 置信度必须 `detach()`。

---

# 9. Student 对抗损失

Student 希望其特征被判别器视为专家特征：

\[
\ell_{ik}^{adv}
=
BCE(D_k(F_i^G),1)
\]

统一加权：

\[
\mathcal L_{adv}
=
\frac{
\sum_i\sum_k
w_{ik}\ell_{ik}^{adv}
}{
\sum_i\sum_kw_{ik}+\varepsilon
}
\]

Student 总损失：

\[
\mathcal L_G
=
\mathcal L_{det}
+
\lambda_{adv}\mathcal L_{adv}
+
\text{项目原有其他组件损失}
\]

聚类置信度只允许作用于：

- 对抗损失；
- 风格域分类损失；
- 与伪风格标签相关的特征匹配或蒸馏损失。

禁止作用于：

\[
\mathcal L_{det}
\]

因为风格聚类不确定不等于目标检测标注不可靠。

---

# 10. 为什么必须统一损失归一化

不能直接写成：

```python
loss_adv = loss_adv_0 + loss_adv_1 + loss_adv_2
```

否则：

- `hard` 每张图只有 1 个有效分支；
- `soft_top2` 每张图有 2 个有效分支；
- `soft_dense` 每张图有 3 个有效分支。

即使算法没有变好，三分支模式也可能因损失天然变大而获得更强梯度，导致消融实验不公平。

必须使用加权平均：

```python
loss = weighted_loss_sum / (weight_sum + eps)
```

对于不使用置信度的模式，每张图片路由权重和为 1，因此 batch 的总权重约为 batch size。

对于使用置信度的模式，分母使用置信度后的有效权重和，保证不同 batch 的整体损失尺度稳定。

---

# 11. 建议的项目文件结构

应先根据项目现有结构调整位置，不要求强制使用以下目录名，但职责必须拆分清楚。

```text
yolov8-pytorch-master/
├── dg_config.py
├── train_5datasets.py
├── train_style_ablation.py
│
├── tools/
│   └── build_style_routing_cache.py
│
├── utils/
│   ├── style_assignment.py
│   ├── style_router.py
│   ├── style_adv_trainer.py
│   └── dataloader.py
│
├── style_cache/
│   ├── train_style_distances.npz
│   ├── val_style_distances.npz
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   └── cluster_metadata.json
│
└── model_data/
    ├── style_0_best.pth
    ├── style_1_best.pth
    └── style_2_best.pth
```

如果项目已有组件目录，应优先放入已有目录，不要重复建立同职责模块。

---

# 12. `dg_config.py` 设计

## 12.1 主实验预设

不要只依赖多个互相独立的布尔开关，因为容易出现非法组合。

推荐增加：

```python
STYLE_EXPERIMENT_PRESETS = {
    "style_hard": {
        "mode": "hard",
        "topk": 1,
        "use_confidence": False,
    },
    "exp3_soft_dense": {
        "mode": "soft_dense",
        "topk": 3,
        "use_confidence": False,
    },
    "exp4_soft_top2": {
        "mode": "soft_top2",
        "topk": 2,
        "use_confidence": False,
    },
    "exp5_soft_top2_conf": {
        "mode": "soft_top2_conf",
        "topk": 2,
        "use_confidence": True,
    },
}

STYLE_EXPERIMENT_NAME = "exp3_soft_dense"
```

用户后续切换实验时，原则上只修改：

```python
STYLE_EXPERIMENT_NAME
```

---

## 12.2 推荐配置项

将以下配置以项目当前 `dg_config.py` 的格式加入。若项目当前使用类、字典或模块级变量，应保持一致。

```python
STYLE_ADV_ENABLED = True
STYLE_EXPERIMENT_NAME = "exp3_soft_dense"
NUM_STYLE_DOMAINS = 3

STYLE_EXPERT_PATHS = [
    "model_data/style_0_best.pth",
    "model_data/style_1_best.pth",
    "model_data/style_2_best.pth",
]

STYLE_FEATURE_LAYER = "feat1"

FREEZE_STYLE_EXPERTS = True
STYLE_EXPERT_EVAL_MODE = True
STYLE_EXPERT_NO_GRAD = True

STYLE_TRAIN_CACHE = "style_cache/train_style_distances.npz"
STYLE_VAL_CACHE = "style_cache/val_style_distances.npz"

STYLE_SCALER_PATH = "style_cache/scaler.joblib"
STYLE_KMEANS_PATH = "style_cache/kmeans.joblib"
STYLE_CLUSTER_METADATA = "style_cache/cluster_metadata.json"

ASSERT_HARD_LABEL_MATCH = True
HARD_LABEL_MATCH_TOLERANCE = 0.0

STYLE_SOFT_TEMPERATURE = 1.0
STYLE_DISTANCE_POWER = 2
STYLE_SOFT_EPS = 1e-8
STYLE_DISTANCE_SCALE_MODE = "train_median_min_d2"

STYLE_ROUTING_RENORMALIZE = True
STYLE_SPARSE_EXPERT_FORWARD = False
STYLE_MIN_ROUTING_WEIGHT = 0.0

STYLE_CONFIDENCE_TYPE = "entropy"
STYLE_CONFIDENCE_FLOOR = 0.2
STYLE_CONFIDENCE_POWER = 1.0
STYLE_CONFIDENCE_THRESHOLD = None
CONFIDENCE_FROM_FULL_PROB = True

STYLE_ADV_LOSS_WEIGHT = 0.1
STYLE_ADV_REDUCTION = "global_weighted_mean"

STYLE_ADV_WARMUP_ENABLED = True
STYLE_ADV_WARMUP_EPOCHS = 10

DISCRIMINATOR_STEPS = 1
GENERATOR_STEPS = 1

DETACH_ROUTING_WEIGHT = True
DETACH_CONFIDENCE_WEIGHT = True

LOG_STYLE_BRANCH_LOSS = True
LOG_STYLE_BRANCH_MASS = True
LOG_STYLE_PROBABILITY_STATS = True
LOG_STYLE_CONFIDENCE_STATS = True
SAVE_STYLE_ROUTING_SNAPSHOT = True
```

`STYLE_ADV_LOSS_WEIGHT` 默认值应优先使用当前硬聚类对抗实验中已经验证过的值，而不是机械使用 `0.1`。

---

## 12.3 配置解析函数

```python
def resolve_style_experiment_config():
    if STYLE_EXPERIMENT_NAME not in STYLE_EXPERIMENT_PRESETS:
        raise ValueError(
            f"Unknown STYLE_EXPERIMENT_NAME: {STYLE_EXPERIMENT_NAME}"
        )

    preset = STYLE_EXPERIMENT_PRESETS[STYLE_EXPERIMENT_NAME]

    return {
        "experiment_name": STYLE_EXPERIMENT_NAME,
        "mode": preset["mode"],
        "topk": preset["topk"],
        "use_confidence": preset["use_confidence"],
        "temperature": STYLE_SOFT_TEMPERATURE,
        "confidence_type": STYLE_CONFIDENCE_TYPE,
        "confidence_floor": STYLE_CONFIDENCE_FLOOR,
        "confidence_power": STYLE_CONFIDENCE_POWER,
        "adv_weight": STYLE_ADV_LOSS_WEIGHT,
        "sparse_expert_forward": STYLE_SPARSE_EXPERT_FORWARD,
    }
```

---

## 12.4 配置合法性检查

```python
def validate_style_config(cfg):
    valid_modes = {
        "hard",
        "soft_dense",
        "soft_top2",
        "soft_top2_conf",
    }

    if cfg["mode"] not in valid_modes:
        raise ValueError(f"Invalid style mode: {cfg['mode']}")

    if NUM_STYLE_DOMAINS != 3:
        raise ValueError(
            "This implementation currently requires NUM_STYLE_DOMAINS == 3."
        )

    if cfg["temperature"] <= 0:
        raise ValueError("STYLE_SOFT_TEMPERATURE must be > 0.")

    if not 0.0 <= cfg["confidence_floor"] <= 1.0:
        raise ValueError(
            "STYLE_CONFIDENCE_FLOOR must be in [0, 1]."
        )

    if cfg["mode"] == "hard":
        assert cfg["topk"] == 1
        assert not cfg["use_confidence"]

    elif cfg["mode"] == "soft_dense":
        assert cfg["topk"] == 3
        assert not cfg["use_confidence"]

    elif cfg["mode"] == "soft_top2":
        assert cfg["topk"] == 2
        assert not cfg["use_confidence"]

    elif cfg["mode"] == "soft_top2_conf":
        assert cfg["topk"] == 2
        assert cfg["use_confidence"]
```

---

# 13. 离线距离缓存生成

新增或扩展聚类脚本，生成后续实验共同使用的缓存。

## 13.1 输入

训练集：

```text
India_train.txt
Japan_train.txt
USA_train.txt
```

验证集：

```text
India_val.txt
Japan_val.txt
USA_val.txt
```

必须使用硬聚类实验相同的：

- `yolov8_s.pth`
- 图像输入尺寸
- 图像预处理
- `feat1`
- 通道均值计算方式
- 通道标准差计算方式
- scaler
- K-means 模型
- 聚类中心编号

---

## 13.2 推荐缓存字段

`train_style_distances.npz` 与 `val_style_distances.npz` 至少保存：

```python
image_paths:     shape [N]
hard_labels:    shape [N]
distance_sq:    shape [N, 3]
countries:      shape [N]
splits:         shape [N]
```

不要只保存 soft probability，因为后续调整温度时需要重新计算概率。

---

## 13.3 `cluster_metadata.json`

建议保存：

```json
{
  "num_clusters": 3,
  "feature_layer": "feat1",
  "style_vector": "channel_mean_std",
  "input_shape": [640, 640],
  "distance_type": "squared_euclidean",
  "distance_scale": 1.234567,
  "distance_scale_mode": "train_median_min_d2",
  "kmeans_random_state": 0,
  "kmeans_n_init": 20,
  "source_train_files": [
    "India_train.txt",
    "Japan_train.txt",
    "USA_train.txt"
  ],
  "style_expert_paths": [
    "style_0_best.pth",
    "style_1_best.pth",
    "style_2_best.pth"
  ]
}
```

需要保存实际参数，不要固定照抄示例值。

---

## 13.4 路径规范化

```python
import os

def canonicalize_path(path: str) -> str:
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(path.strip())
        )
    )
```

如果路径无法匹配，必须报错，禁止静默使用默认标签或默认权重。

---

# 14. `style_assignment.py`

建议实现：

```python
from dataclasses import dataclass
import torch


@dataclass
class StyleAssignmentOutput:
    full_prob: torch.Tensor
    hard_from_prob: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    confidence: torch.Tensor
```

核心接口：

```python
def compute_soft_assignment(
    distance_sq: torch.Tensor,
    temperature: float,
    distance_scale: float,
    eps: float = 1e-8,
    confidence_floor: float = 0.2,
    confidence_power: float = 1.0,
) -> StyleAssignmentOutput:
    pass
```

实现逻辑：

```python
scaled_logits = -distance_sq / (
    temperature * distance_scale + eps
)

full_prob = torch.softmax(scaled_logits, dim=1)
hard_from_prob = torch.argmax(full_prob, dim=1)

entropy = -torch.sum(
    full_prob * torch.log(full_prob + eps),
    dim=1,
)

log_k = torch.log(
    torch.tensor(
        full_prob.shape[1],
        device=full_prob.device,
        dtype=full_prob.dtype,
    )
)

normalized_entropy = entropy / log_k
raw_confidence = 1.0 - normalized_entropy

confidence = (
    confidence_floor
    + (1.0 - confidence_floor) * raw_confidence
)

confidence = confidence.pow(confidence_power)
```

---

# 15. `style_router.py`

## 15.1 输出结构

```python
@dataclass
class StyleRoutingOutput:
    full_prob: torch.Tensor
    routing_prob: torch.Tensor
    confidence: torch.Tensor
    final_weight: torch.Tensor
    hard_style: torch.Tensor
    active_mask: torch.Tensor
```

形状：

```text
full_prob:     [B, 3]
routing_prob:  [B, 3]
confidence:    [B]
final_weight:  [B, 3]
hard_style:    [B]
active_mask:   [B, 3]
```

---

## 15.2 Top-K

```python
def topk_normalize(
    prob: torch.Tensor,
    topk: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    values, indices = torch.topk(
        prob,
        k=topk,
        dim=1,
    )

    sparse_prob = torch.zeros_like(prob)
    sparse_prob.scatter_(
        dim=1,
        index=indices,
        src=values,
    )

    return sparse_prob / (
        sparse_prob.sum(dim=1, keepdim=True) + eps
    )
```

---

## 15.3 统一路由

```python
def build_style_routing(
    hard_style: torch.Tensor,
    distance_sq: torch.Tensor,
    mode: str,
    topk: int,
    temperature: float,
    distance_scale: float,
    confidence_floor: float,
    confidence_power: float,
    eps: float = 1e-8,
) -> StyleRoutingOutput:
    assignment = compute_soft_assignment(
        distance_sq=distance_sq,
        temperature=temperature,
        distance_scale=distance_scale,
        eps=eps,
        confidence_floor=confidence_floor,
        confidence_power=confidence_power,
    )

    full_prob = assignment.full_prob
    confidence = assignment.confidence

    if mode == "hard":
        routing_prob = torch.nn.functional.one_hot(
            hard_style,
            num_classes=full_prob.shape[1],
        ).to(full_prob.dtype)
        confidence = torch.ones_like(confidence)

    elif mode == "soft_dense":
        routing_prob = full_prob
        confidence = torch.ones_like(confidence)

    elif mode == "soft_top2":
        routing_prob = topk_normalize(full_prob, topk=2)
        confidence = torch.ones_like(confidence)

    elif mode == "soft_top2_conf":
        routing_prob = topk_normalize(full_prob, topk=2)

    else:
        raise ValueError(
            f"Unsupported style routing mode: {mode}"
        )

    final_weight = routing_prob * confidence.unsqueeze(1)
    active_mask = routing_prob > 0

    return StyleRoutingOutput(
        full_prob=full_prob.detach(),
        routing_prob=routing_prob.detach(),
        confidence=confidence.detach(),
        final_weight=final_weight.detach(),
        hard_style=assignment.hard_from_prob.detach(),
        active_mask=active_mask.detach(),
    )
```

---

# 16. 数据集与 `collate_fn`

主训练数据仍覆盖三个源域全部图片。

每个样本至少返回：

```python
image
targets
image_path
hard_style
distance_sq
```

`collate_fn` 应将：

```text
hard_style → LongTensor [B]
distance_sq → FloatTensor [B, 3]
```

不要在数据集内部固定计算最终软概率，避免修改温度时重新生成数据。

---

# 17. 三个风格专家加载

必须保证：

```text
q0 → Style_0 专家 → D0
q1 → Style_1 专家 → D1
q2 → Style_2 专家 → D2
```

加载后：

```python
for expert in style_experts:
    expert.eval()
    for param in expert.parameters():
        param.requires_grad = False
```

专家前向：

```python
with torch.no_grad():
    expert_feature = expert(images)
```

专家和 Student 必须提取同一对抗位置、形状兼容的特征。

---

# 18. 训练循环接入

优先在 `train_5datasets.py` 的原组件位置接入。若代码过于复杂，可复制为 `train_style_ablation.py`，但必须保持其他训练条件一致。

## 18.1 Student 前向

```python
student_outputs, student_feat = student_model(images)
loss_det = yolo_loss(student_outputs, targets)
```

具体接口按项目现有代码适配。

---

## 18.2 路由

```python
route = build_style_routing(
    hard_style=hard_style,
    distance_sq=distance_sq,
    mode=style_cfg["mode"],
    topk=style_cfg["topk"],
    temperature=style_cfg["temperature"],
    distance_scale=distance_scale,
    confidence_floor=style_cfg["confidence_floor"],
    confidence_power=style_cfg["confidence_power"],
)
```

必须检查：

```python
match = route.hard_style.eq(hard_style)
```

默认不一致即终止。

---

## 18.3 第一版专家前向

第一版三个专家均处理完整 batch：

```python
expert_features = []

with torch.no_grad():
    for expert in style_experts:
        feat = expert(images)
        expert_features.append(feat)
```

这时 Top-2 主要是损失路由稀疏，不一定显著减少专家前向成本。

---

## 18.4 更新判别器

```python
set_requires_grad(discriminators, True)
optimizer_d.zero_grad()

loss_d, loss_d_detail = compute_weighted_discriminator_loss(
    student_feat=student_feat.detach(),
    expert_features=expert_features,
    discriminators=discriminators,
    branch_weight=route.final_weight.detach(),
)

loss_d.backward()
optimizer_d.step()
```

---

## 18.5 更新 Student

```python
set_requires_grad(discriminators, False)
optimizer_g.zero_grad()

loss_adv, loss_adv_detail = compute_weighted_generator_adv_loss(
    student_feat=student_feat,
    discriminators=discriminators,
    branch_weight=route.final_weight.detach(),
)

loss_total = (
    loss_det
    + current_adv_weight * loss_adv
    + existing_component_losses
)

loss_total.backward()
optimizer_g.step()
```

若原项目使用 GRL，应保持原对抗更新机制，仅将逐样本分支权重接入原损失。

---

# 19. 加权损失要求

每个判别器必须输出逐样本损失，不能先 batch 平均再乘平均权重。

错误：

```python
loss_k = criterion(pred, label).mean()
loss_k = loss_k * branch_weight[:, k].mean()
```

正确：

```python
raw_loss = criterion_without_batch_reduction(
    pred,
    label,
)

per_sample_loss = raw_loss.flatten(1).mean(dim=1)
weight_k = branch_weight[:, k]

weighted_sum_k = (
    per_sample_loss * weight_k
).sum()
```

最后统一：

```python
total_loss = total_weighted_sum / (
    total_weight + eps
)
```

---

# 20. 对抗 warmup

如原项目没有现成实现，可增加：

```python
def get_adv_warmup_weight(
    base_weight,
    epoch,
    warmup_epochs,
):
    if not STYLE_ADV_WARMUP_ENABLED:
        return base_weight

    if warmup_epochs <= 0:
        return base_weight

    ratio = min(
        1.0,
        max(0.0, epoch / float(warmup_epochs))
    )

    return base_weight * ratio
```

---

# 21. 稀疏专家前向（二阶段优化）

只有第一版逻辑验证正确后，再实现：

```python
STYLE_SPARSE_EXPERT_FORWARD = True
```

示意：

```python
for k in range(3):
    mask_k = route.active_mask[:, k]

    if mask_k.any():
        images_k = images[mask_k]

        with torch.no_grad():
            expert_feat_k = style_experts[k](images_k)

        student_feat_k = student_feat[mask_k]
        weight_k = route.final_weight[mask_k, k]
```

注意 DDP 空分支、BatchNorm、子 batch 过小和未使用参数问题。默认保持 `False`。

---

# 22. 多 GPU / DDP 注意事项

1. 首版建议三个判别器完整前向，通过权重置零。
2. 某个 rank 可能没有某分支样本。
3. 空分支不要返回 Python 数字 `0`。
4. 可返回：
   ```python
   zero_loss = student_feat.sum() * 0.0
   ```
5. 全局路由统计需进行 `all_reduce`。
6. 保持项目原有 sampler 和随机种子逻辑。

---

# 23. 日志与实验目录

自动目录示例：

```text
logs/
└── exp5_soft_top2_conf_tau1.0_floor0.2_adv0.1/
```

每次训练保存：

```text
config_snapshot.py
style_experiment_config.json
cluster_metadata.json
train_log.csv
routing_statistics.csv
best_epoch_weights.pth
last_epoch_weights.pth
```

必须记录：

```text
hard label match rate
mean(max(q))
mean entropy
mean normalized entropy
mean confidence
Style_0 branch mass
Style_1 branch mass
Style_2 branch mass
loss_d0
loss_d1
loss_d2
loss_adv0
loss_adv1
loss_adv2
```

Top-2 还需记录三种组合数量。

---

# 24. 推荐消融实验

| 实验编号 | 配置名称 | 模式 | Top-K | 置信度 |
|---|---|---|---:|---|
| E2 | `style_hard` | hard | 1 | 无 |
| E3 | `exp3_soft_dense` | soft_dense | 3 | 无 |
| E4 | `exp4_soft_top2` | soft_top2 | 2 | 无 |
| E5 | `exp5_soft_top2_conf` | soft_top2_conf | 2 | 熵 |

保持其他训练条件完全一致。

温度建议：

```python
0.5
1.0
2.0
```

置信度下限建议：

```python
0.0
0.1
0.2
0.5
```

---

# 25. 启动配置摘要

训练开始时输出：

```text
[Style Experiment]
name: exp5_soft_top2_conf
mode: soft_top2_conf
num_style_domains: 3
topk: 2
temperature: 1.0
confidence_enabled: True
confidence_type: entropy
confidence_floor: 0.2
confidence_power: 1.0
adv_loss_weight: 0.1
sparse_expert_forward: False
train_cache: ...
style_expert_0: ...
style_expert_1: ...
style_expert_2: ...
distance_scale: ...
hard_label_match_rate: 1.000000
```

---

# 26. 单元测试

建议新增：

```text
tests/test_style_router.py
```

至少验证：

1. `argmax(q) == argmin(distance_sq)`。
2. Top-2 每行仅两个非零权重。
3. Top-2 每行权重和为 1。
4. hard 模式等于 one-hot。
5. 置信度位于 `[floor, 1]`。
6. `[0.90,0.07,0.03]` 的置信度高于 `[0.34,0.33,0.33]`。
7. hard、Top-2、Top-3 不会仅因分支数量不同导致损失机械变成 1、2、3 倍。
8. 固定随机输入可完成一次判别器反向与 Student 反向。

---

# 27. 验收标准

1. 仅修改 `STYLE_EXPERIMENT_NAME` 即可运行四种模式。
2. `style_hard` 能复现原硬聚类流程。
3. `argmax(soft_prob)` 与原硬标签一致率为 100%。
4. 三个专家、三个判别器编号严格对应。
5. 聚类置信度不作用于检测损失。
6. 分支损失使用逐样本加权和全局归一化。
7. 所有路由权重默认 detach。
8. 日志包含完整路由统计。
9. 不同实验不会覆盖同一日志目录。
10. 训练与验证均能完成小 batch 测试。

---

# 28. 推荐实施顺序

## 阶段 1：检查原项目

1. 阅读 `三组件详解.txt`。
2. 阅读 `dg_config.py`。
3. 阅读 `train_5datasets.py`。
4. 找到原硬风格对抗入口。
5. 确认三个专家和三个判别器对应关系。
6. 确认原损失和更新方式。

## 阶段 2：距离缓存

1. 复用现有 scaler 和 K-means。
2. 生成训练、验证距离缓存。
3. 保存 metadata。
4. 检查标签一致率。

## 阶段 3：路由模块

1. soft assignment。
2. hard one-hot。
3. soft dense。
4. Top-2。
5. 熵置信度。
6. 单元测试。

## 阶段 4：训练接入

1. 数据集返回距离。
2. 训练循环构建路由。
3. 加权判别器损失。
4. 加权 Student 对抗损失。
5. 检测损失保持不变。

## 阶段 5：配置和日志

1. 实验预设。
2. 参数合法性检查。
3. 自动目录。
4. 配置快照。
5. 路由统计。

## 阶段 6：稀疏前向

在算法正确后再实现专家子 batch 加速。

---

# 29. Codex 修改约束

1. 不删除原有组件。
2. 不破坏原训练入口。
3. 不修改检测标注 txt 格式。
4. 不把软权重追加到检测标注行。
5. 不在训练时重新拟合 scaler 或 K-means。
6. 不在验证集拟合聚类参数。
7. 不重新训练三个风格专家。
8. 不改变聚类中心编号和专家编号对应关系。
9. 不使用聚类置信度降低检测损失。
10. 不让离线路由权重参与梯度。
11. 不无归一化地直接相加三个分支损失。
12. hard 模式未复现前，不进入正式软聚类实验。
13. 新增函数必须注明输入输出 shape。
14. 项目实际接口与本文示例不同，应适配现有实现，不要创建重复职责模块。

---

# 30. 最终交付内容

Codex 完成后应给出：

1. 修改文件清单。
2. 每个文件的修改说明。
3. 新增配置项说明。
4. 四种模式的运行方式。
5. 缓存生成方式。
6. 标签一致率检查结果。
7. 单元测试结果。
8. 小 batch 前向与反向结果。
9. 日志目录示例。
10. 尚未实现或存在风险的部分。

---

# 31. 最终使用方式

完成开发后，用户只修改：

```python
STYLE_EXPERIMENT_NAME = "style_hard"
```

或：

```python
STYLE_EXPERIMENT_NAME = "exp3_soft_dense"
```

或：

```python
STYLE_EXPERIMENT_NAME = "exp4_soft_top2"
```

或：

```python
STYLE_EXPERIMENT_NAME = "exp5_soft_top2_conf"
```

进一步调参只修改：

```python
STYLE_SOFT_TEMPERATURE
STYLE_CONFIDENCE_FLOOR
STYLE_CONFIDENCE_POWER
STYLE_ADV_LOSS_WEIGHT
STYLE_ADV_WARMUP_EPOCHS
STYLE_SPARSE_EXPERT_FORWARD
```

无需修改核心训练代码。

---

# 32. 总结

本任务是在现有风格硬聚类对抗框架中，将“每张图片只选择一个风格专家和判别器”的硬路由，扩展为：

1. 完整软风格路由；
2. Top-2 稀疏风格路由；
3. 置信度感知的 Top-2 路由。

四种模式共用现有三个风格专家、三个判别器和同一训练入口，并通过 `dg_config.py` 完成切换、调参和消融实验。
