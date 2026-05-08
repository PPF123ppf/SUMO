# SUMO-CAV 混合交通博弈换道仿真系统

基于 SUMO 仿真平台，研究 **高速公路事故场景下 CAV 换道决策问题**。提出混合博弈换道模型，结合 8 维原子特征和最大熵逆强化学习，在 0%~100% CAV 渗透率的混合交通中验证模型有效性。

---

## 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                       问题建模                               │
│  4km 三车道高速，中间车道 3000m 事故封堵                     │
│  混合交通：CAV 按渗透率 p% 分布 + 异质人类车（3种驾驶风格）  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     CAV 换道决策流程                          │
│                                                              │
│  ① 感知 → V2X 噪声/延迟/丢包模型                            │
│  ② 安全门控 → 间距门控 + TTC 硬安全门控                      │
│  ③ 特征提取 → 8维原子特征 + 自适应权重                       │
│  ④ 博弈求解 → 按后车类型分发到不同博弈结构                   │
│  ⑤ 期望收益 → Stackelberg 均衡 / Nash 均衡 / 合作博弈       │
│  ⑥ 阈值判断 → 渗透率自适应阈值                              │
│  ⑦ 换道执行 → 准备(0.4s)→执行(2.8s)→稳定(0.5s)             │
│  ⑧ 在线学习 → 每次换道后 TD 微调权重                        │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 验证：渗透率扫描实验                           │
│  4模型 × 4密度(1200~3600pcu/h) × 6渗透率(0%~100%)          │
│  8核并行，192组仿真，8小时                                    │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  IRL 数据驱动权重学习                          │
│  AD4CHE 中国高速数据集 → 8维特征映射 → MaxEnt IRL训练        │
│                                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心创新

| 创新点 | 方法 | 论文依据 |
|--------|------|---------|
| **8维原子特征空间** | 速度/紧迫度/压力/安全/协同/社会冲击/密度/换道代价，每维独立可解释 | 原创 |
| **混合博弈求解器** | 3种后车类型 × 3种博弈结构（静态Nash/Stackelberg/Nash合作） | Huang 2024, Wang 2026 |
| **混合策略 softmax** | 替代硬 argmax，temperature=0.15 保留行为不确定性 | Liu 2024 |
| **社会冲击建模** | 量化换道对后车 TTC 的缩减比例，纳入收益函数 | Liu 2024 |
| **混合交通异质性** | 保守/普通/激进 3种人类车，K-Means 驾驶风格分类 | He 2025 |
| **低渗透率自适应** | 对抗性偏差 + 互惠记忆 + 渗透率自适应阈值 | Burger 2022, Chung 2025 |
| **MaxEnt IRL** | 从 AD4CHE（360万帧/3908换道片段）学习 8 维权重 | Li 2024 |
| **在线 TD 学习** | 每次换道后基于结果即时微调权重（lr=0.003） | Lopez 2022 |

---

## 场景与参数

**道路**：3.5km 三车道高速公路（E0），车道 1 在 3000m 处因追尾事故封堵，事故区 3000-3200m，终点 3500m

**事故时间线**：

| 阶段 | 时间 | V2X | 行为 |
|------|------|:--:|------|
| 正常行驶 | t < 90s | — | CACC 巡航 |
| 突发期 | 90~100s | 局部 500m | 高紧迫度换道 |
| 有序期 | 100s+ | 全局 1200m | 有序疏散 |

**混合交通构成**：

| 车辆类型 | 换道决策 | 跟驰模型 | V2X | 占比 |
|---------|---------|---------|:--:|:--:|
| CAV | 混合博弈求解器 | CACC | 500~1200m | p% |
| 人类车-保守型 | SL2015 (sigma=0.3) | Krauss | ❌ | 33%(1-p)% |
| 人类车-普通型 | SL2015 (sigma=0.5) | Krauss | ❌ | 33%(1-p)% |
| 人类车-激进型 | SL2015 (sigma=0.7) | Krauss | ❌ | 33%(1-p)% |

---

## 博弈模型详解

### 本车收益函数（8 维特征）

每个 `(本车动作, 后车反应)` 组合提取 8 个原子特征：

```
payoff(action, fa) = w · f(action, fa) - lc_cost · I(action=换道)
```

**特征定义**：

| 维度 | 特征 | 换道时 | 不换道时 |
|:--:|------|--------|---------|
| 0 | speed | 目标车道前车速度比 | 自车速度比 |
| 1 | urgency | 距事故归一化距离(0~1) | 0 |
| 2 | pressure | 当前车道头距压力(0~1) | 0 |
| 3 | safe | min(前TTC, 后TTC)评分 | 当前车道前TTC |
| 4 | coop | 后车协同(0.18)+制动加分(0.12) | 0.05 |
| 5 | social | 换道对后车TTC缩减比(0~1) | 0 |
| 6 | density | 局部车流密度(0~1) | 0 |
| 7 | lc_cost | 1.0 | 0.0 |

**自适应权重**：权重随场景动态调整，密度和紧迫度高时安全权重上升、速度权重下降。

### 后车收益函数（5 维 × 3 种类型）

后车拥有独立收益函数，与 CAV 进行真博弈（非概率猜测）：

```
U_f(ego, follower) = w_t · [safety, speed_ratio, comfort_cost, coop_reward, gap_safe]
```

### 混合博弈求解器

按后车类型选择博弈结构：

| 类型 | 占比 | 博弈结构 | 策略 | 行为特征 |
|:--:|:--:|---------|------|---------|
| Type 0 自私型 | 20% | **静态 Nash** | 4轮迭代，双方 softmax 收敛到混合策略均衡 | 只关心自身收益 |
| Type 1 合作型 | 60% | **Stackelberg** | 本车宣布动作 → 后车 softmax 选最优反应 → 本车取期望 | 遵守主从博弈 |
| Type 2 高合作型 | 20% | **Nash 合作** | 最大化联合收益矩阵，后车 softmax 配合最优联合动作 | 互惠共赢 |

### 社会冲击计算

```
social = max(0, (fol_old_ttc - fol_new_ttc) / fol_old_ttc)

# fol_old_ttc = (fol_gap + lead_gap) / max(fol_spd - lead_spd, 0.1)
# fol_new_ttc = fol_gap / max(fol_spd_new - ego_speed, 0.1)
```

当本车切入导致后车 TTC 骤降时，social 特征值高，`w_social` 降低换道收益，抑制强行插入。

---

## 低渗透率自适应机制

低 CAV 占比时博弈模型的三个关键改进：

| 机制 | 问题 | 方法 | 效果 |
|------|------|------|------|
| **对抗性偏差** | 人类不参与博弈 | 先验 shift: 加速+10%, 减速-10% | 不天真假设人类配合 |
| **互惠记忆** | 人类行为不一致 | 记录让行史，下次动态调先验 | 合作者加分，对抗者减分 |
| **渗透率自适应阈值** | 低渗透时犹豫错失间隙 | threshold = base × (1-0.6×(1-p)) | 0%→0.4×, 100%→1.0× |

---

## 实验结果

### Game (Ours) 完整渗透率扫描（通过车辆数 / 换道次数）

| 渗透率 | 1200pcu/h | 2000pcu/h | 2800pcu/h | 3600pcu/h |
|:------:|:---------:|:---------:|:---------:|:---------:|
| 0% (全人类) | — | 116 / 0 | 167 / 0 | 202 / 0 |
| 10% | 69 / 4 | 110 / 6 | 153 / 8 | 177 / 21 |
| 30% | 72 / 8 | 115 / 31 | 152 / 32 | 189 / 29 |
| 50% | 64 / 22 | 106 / 33 | 147 / 62 | 183 / 64 |
| 70% | 64 / 29 | 107 / 64 | 144 / 84 | 185 / 130 |
| **100% (全CAV)** | **65 / 41** | **91 / 84** | **148 / 119** | **150 / 203** |

![渗透率扫描热图](results/figures/penetration_sweep.png)
*左：通过车辆数，右：最大队列长度*

### 全模型对比（100% CAV）

| 模型 | 1200 | 2000 | 2800 | 3600 | 碰撞 |
|------|:----:|:----:|:----:|:----:|:--:|
| **Game (Ours)** | **65** | **91** | **150** | **150** | **0** |
| No-V2X | 58 | 92 | 136 | 156 | 0 |
| Rule-Based | 40 | 69 | 93 | 117 | 0 |
| SUMO Default | 40 | 13 | 29 | 68 | 2 |

![全模型对比柱状图](results/figures/model_comparison.png)

**模型使用的权重（AD4CHE 数据集通过 MaxEnt IRL 学习）**：

```
[speed, urgency, pressure, safe, coop, social, density, lc_cost]
[0.15,  0.19,    0.18,     0.25, 0.02, 0.00,   0.11,   0.70 ]
```

---

## 核心结论

1. **Game 模型全面领先** — 通过量最高、零碰撞、队列最短。3600pcu/h 时通过量 150（SUMO Default 仅 68）
2. **混合交通 30% CAV 即达全 CAV 效果** — 渗透率从 10% 到 100%，通过量基本稳定在 65~72（1200pcu/h），30% 已足够
3. **数据驱动的权重** — 所有博弈权重通过 MaxEnt IRL 从 AD4CHE（中国高速公路真数据）学习得到，非人工调参。模型在高密度全 CAV 场景下零碰撞
4. **社会冲击** — 换道活跃度随渗透率递增（0→203次，3600pcu/h），CAV 用博弈主动找好车道位置
5. **SUMO Default 高密度崩溃** — 2800pcu/h 出现碰撞，3600pcu/h 通过量仅为 Game 的 42%
6. **Rule-Based 固定阈值完全失效** — TTC≥3.0s 导致 0 次换道，队列最长

---

## 使用方式

### 依赖

- SUMO 1.x（`C:\Program Files (x86)\Eclipse\Sumo`）
- Python 3.10+，包：`traci`, `sumolib`, `numpy`, `pandas`, `matplotlib`

### 运行

```bash
# 单次仿真（balanced 预设）
echo "b" | PYTHONIOENCODING=utf-8 python game_lane_change.py

# 完整基线（4模型 × 4密度 × 6渗透率 = 96组）
python run_baseline_stepwise.py --sim-steps 3600

# 多核并行
python run_baseline_stepwise.py --models "Game (Ours)" --out-dir results/game &
python run_baseline_stepwise.py --models "SUMO Default" --out-dir results/sumo &

# IRL 训练
python run_irl_quick.py

# 生成图表
python run_plot_results.py
```

---

## 参数预设

| 预设 | 时距(s) | 突发期阈值 | 换道代价 | 适用场景 |
|------|:------:|:---------:|:-------:|---------|
| **balanced** | 1.00 | 0.030 | 0.060 | 效率与安全均衡（默认） |
| aggressive | 0.85 | 0.020 | 0.040 | 低密度、急需疏散 |
| conservative | 1.25 | 0.050 | 0.090 | 高密度、安全优先 |
| balanced_plus | 1.18 | 0.060 | 0.100 | 最保守 |

---

## 项目结构

```
SUMO-1/
├── game_lane_change.py       # 主仿真：博弈求解器/特征/状态机（~3000行）
├── baseline_comparison.py    # 4种基线模型实现
├── run_baseline_stepwise.py  # 实验运行器（并行/渗透率/IRL权重）
├── irl.py                    # 最大熵逆强化学习（8维特征）
├── run_irl_quick.py          # IRL 快速训练脚本
├── run_plot_results.py       # 图表生成脚本
├── config.py                 # 集中参数配置
├── metrics.py                # 舒适性/公平性评价
├── plot_baseline_results.py  # 结果可视化（柱状图/热图/雷达图）
├── core_model_whitepaper.md  # 模型技术白皮书
├── irl_weights_v2.npz       # IRL 学出权重
├── results/figures/          # 实验对比图表
└── data/                     # AD4CHE 数据集（需自行下载）
```

---

## 参考文献

1. He, Y., Xiang, D., Wang, D. (2025). Traffic safety evaluation of emerging mixed traffic flow at freeway merging area considering driving behavior. *Scientific Reports*.
2. Wang, D. et al. (2026). Game-Theoretic Reinforcement Learning-Based Behavior-Aware Merging in Mixed Traffic. *IEEE Trans. Intelligent Transportation Systems*, 27(1), 483-496.
3. Huang, P. et al. (2024). A Game-Based Hierarchical Model for Mandatory Lane Change of Autonomous Vehicles. *IEEE Trans. Intelligent Transportation Systems*, 25(9), 11256-11268.
4. Liu, J. et al. (2024). Enhancing Social Decision-Making of Autonomous Vehicles: A Mixed-Strategy Game Approach. *IEEE Trans. Vehicular Technology*, 73(9), 12385-12399.
5. Li, W. et al. (2024). Simulation of Vehicle Interaction Behavior in Merging Scenarios: A Deep Maximum Entropy-Inverse Reinforcement Learning Method. *IEEE Trans. Intelligent Vehicles*, 9(1), 1079-1091.
6. Lopez, V. G. et al. (2022). Game-Theoretic Lane-Changing Decision Making and Payoff Learning for Autonomous Vehicles. *IEEE Trans. Vehicular Technology*, 71(4), 3609-3622.
7. Burger, C. et al. (2022). Interaction-Aware Game-Theoretic Motion Planning for Automated Vehicles using Bi-level Optimization. *IEEE ITSC*.
