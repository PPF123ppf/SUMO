# SUMO-CAV 混合交通博弈换道仿真

基于 SUMO 的高速公路事故场景仿真，提出**混合博弈换道决策模型**——8 维原子特征 + 3 种博弈结构自适应 + MaxEnt IRL 数据驱动权重学习。支持 0%~100% CAV 渗透率的混合交通实验，通过 4 种基线模型的逐层消融实验评估模型性能。

---

## 技术路线

```
道路: 3.5km 三车道高速, 3000m 事故封堵
车辆: CAV (p%) + 异质人类车 (1-p%)
      ↓
CAV 换道决策:
  ① V2X 感知 (噪声/延迟/丢包)
  ② 安全门控 (间距 + TTC)
  ③ 8 维特征提取 + 自适应权重
  ④ 混合博弈求解 (3 种博弈结构自适应选择)
  ⑤ 渗透率自适应阈值判断
  ⑥ 三阶段换道执行 (准备→执行→稳定)
      ↓
验证: 4 模型 × 4 密度 × 6 渗透率 = 96 组实验
      ↓
消融分析: 逐层剥离 V2X→博弈→Python 控制，量化各模块贡献
```

---

## 核心创新

| 创新点 | 方法 | 论文依据 |
|--------|------|---------|
| **混合博弈求解器** | 3 种后车类型(自私/合作/高合作) × 3 种博弈结构(静态Nash/Stackelberg/Nash合作) | Huang 2024, Wang 2026 |
| **8 维原子特征** | speed/urgency/pressure/safe/coop/social/density/lc_cost，每维可独立解释 | 原创 |
| **Social Impact 建模** | 量化换道对后车 TTC 缩减比，纳入收益函数 | Liu 2024 |
| **混合交通异质性** | 保守/普通/激进 3 种人类车，基于 K-Means 驾驶风格分类 | He 2025 |
| **低渗透率自适应** | 对抗性偏差 + 互惠记忆 + 渗透率自适应阈值 | Burger 2022, Chung 2025 |
| **MaxEnt IRL 权重学习** | 从 AD4CHE（360 万帧/3908 换道片段）学习 8 维权重 | Li 2024 |
| **在线 TD 学习** | 每次换道后基于结果即时微调权重 (lr=0.003) | Lopez 2022 |

---

## 模型架构

### 博弈模型

```
本车收益: U_e(action, fa) = w · f(action, fa) - lc_cost × I(换道)

f = [speed, urgency, pressure, safe, coop, social, density, lc_cost]
w = 自适应权重 (密度/紧迫度 → 安全↑ 速度↓)
```

```
后车收益: U_f(ego, follower) = w_t · [safety, speed, comfort, coop, gap_safe]

Type 0 (20%) 自私型  → 静态Nash    双方同时决策, 混合策略均衡
Type 1 (60%) 合作型  → Stackelberg 本车宣布动作, 后车最优反应
Type 2 (20%) 高合作型 → Nash合作   最大化联合收益
```

### 8 维特征定义

| 维度 | 特征 | 换道时的值 | 不换道时的值 | 物理含义 |
|:--:|------|-----------|-------------|----------|
| 0 | speed | 目标车道前车速度比 | 自车速度比 | 速度收益 |
| 1 | urgency | 距事故归一化距离 | 0 | 紧迫驱动力 |
| 2 | pressure | 当前车道头距压力 | 0 | 逃离压力 |
| 3 | safe | min(前TTC, 后TTC) 安全评分 | 当前车道 TTC | 安全底线 |
| 4 | coop | 后车协同奖励 + 制动加分 | 0.05 | 协同奖励 |
| 5 | social | 换道对后车 TTC 缩减比 | 0 | 社会冲击 |
| 6 | density | 局部车流密度 | 0 | 环境压力 |
| 7 | lc_cost | 1.0 | 0.0 | 换道固有代价 |

IRL 学出权重：`[0.15, 0.19, 0.18, 0.25, 0.02, 0.00, 0.11, 0.70]`

### 低渗透率自适应机制

| 机制 | 问题 | 方法 | 效果 |
|------|------|------|------|
| 对抗性偏差 | 人类不参与博弈 | 先验: 加速+10%, 减速-10% | CAV 不再天真 |
| 互惠记忆 | 人类行为不一致 | 记录让行史, 动态调先验 | 识别合作者 |
| 阈值自适应 | 低渗透犹豫错失间隙 | base × (1-0.6×(1-p)) | 0%→0.4×, 100%→1.0× |

### 三阶段换道执行

```
准备(0.4s) → 安全复查 → 执行(2.8s) → 稳定(0.5s) → 完成
                        ↘ 取消 → online_update_weights()
```

---

## 四模型消融实验

实验设计：通过逐层关闭功能模块（V2X→博弈→Python控制），量化每个模块的贡献。

| 模型 | 换道决策 | V2X 通信 | 协同让行 | 消融目的 |
|------|---------|:--:|:--:|------|
| **Game (Ours)** | 混合博弈 + 8维特征 + 自适应权重 | ✅ 500m/1200m | ✅ | 完整方案 |
| **No-V2X** | 博弈模型 | ❌ | ❌ | 量化 V2X 通信贡献 |
| **Rule-Based** | 多条件规则 (TTC前1.5s+后1.0s) | ❌ | ❌ | 量化博弈 vs 规则 |
| **SUMO Default** | SL2015 原生换道 | ❌ | ❌ | 量化 Python 控制价值 |

### 实验矩阵

4 模型 × 4 密度 (1200/2000/2800/3600 pcu/h) × 6 渗透率 (0%/10%/30%/50%/70%/100%) = 96 组

---

## 实验结果

### Game (Ours) 完整渗透率扫描

通过车辆数 / 换道次数

| 渗透率 | 1200pcu/h | 2000pcu/h | 2800pcu/h | 3600pcu/h |
|:------:|:---------:|:---------:|:---------:|:---------:|
| 0% | 77 / 0 | 109 / 0 | 97 / 0 | 61 / 0 |
| 10% | 77 / 4 | 118 / 6 | 172 / 18 | 214 / 22 |
| 30% | **79** / 8 | **123** / 28 | 159 / 33 | **213** / 37 |
| 50% | 74 / 23 | 122 / 35 | 171 / 61 | 211 / 74 |
| 70% | 78 / 33 | 124 / 58 | 167 / 99 | 214 / 134 |
| **100%** | 77 / 43 | **126** / 82 | **173** / 115 | 204 / 205 |

![渗透率扫描热图](results/figures/penetration_sweep.png)
![渗透率 vs 通过量](results/figures/penetration_throughput.png)
![渗透率 vs 换道活跃度](results/figures/penetration_lc.png)

### 全模型对比 @ 100% CAV

| 模型 | 1200 | 2000 | 2800 | 3600 | 碰撞 |
|------|:----:|:----:|:----:|:----:|:--:|
| **Game (Ours)** | **77** | **126** | **173** | **204** | **0** |
| No-V2X | 77 | 126 | 173 | 203 | **2** |
| Rule-Based | 48 | 80 | 106 | 135 | 0 |
| SUMO Default | 61 | 35 | 41 | 57 | 2 |

![全模型对比](results/figures/model_comparison.png)

---

## 核心结论

1. **Game 模型全面领先** — 2000pcu/h 时通过量 126 辆，SUMO Default 仅 35 辆（**3.6 倍**），零碰撞
2. **渗透率从 10% 起不影响通过量** — 30% CAV 即可获得全部收益，无需 100% 渗透率
3. **V2X 通信贡献在安全，非效率** — Game vs No-V2X 通过量完全一致，但 No-V2X 发生 2 次碰撞，V2X 协同让行是零碰撞的关键
4. **数据驱动权重** — 所有决策权重通过 MaxEnt IRL 从 AD4CHE 数据集学习，无需人工调参
5. **逐层消融验证** — 关闭 V2X (No-V2X) → 效率同、安全降；关闭博弈 (Rule-Based) → 效率降 40%；关闭 Python 控制 (SUMO Default) → 效率降 70%

---

## 使用方式

```bash
# 依赖: SUMO 1.x + Python 3.10+ (traci, sumolib, numpy, pandas, matplotlib)

# 单次仿真
echo "b" | PYTHONIOENCODING=utf-8 python game_lane_change.py

# 完整基线实验 (4模型 × 4密度 × 6渗透率)
python run_baseline_stepwise.py --sim-steps 3600

# 多核并行
python run_baseline_stepwise.py --models "Game (Ours)" --out-dir results/game &
python run_baseline_stepwise.py --models "SUMO Default" --out-dir results/sumo &

# IRL 训练
python run_irl_quick.py

# 图表生成
python run_plot_results.py
```

---

## 参考文献

1. He, Y. et al. (2025). Traffic safety evaluation of emerging mixed traffic flow at freeway merging area. *Scientific Reports*.
2. Wang, D. et al. (2026). Game-Theoretic RL-Based Behavior-Aware Merging in Mixed Traffic. *IEEE Trans. ITS*, 27(1), 483-496.
3. Huang, P. et al. (2024). A Game-Based Hierarchical Model for Mandatory Lane Change of AVs. *IEEE Trans. ITS*, 25(9), 11256-11268.
4. Liu, J. et al. (2024). Enhancing Social Decision-Making of AVs: A Mixed-Strategy Game Approach. *IEEE Trans. VT*, 73(9), 12385-12399.
5. Li, W. et al. (2024). Simulation of Vehicle Interaction Behavior in Merging Scenarios: A Deep MaxEnt-IRL Method. *IEEE Trans. IV*, 9(1), 1079-1091.
6. Lopez, V. G. et al. (2022). Game-Theoretic Lane-Changing Decision Making and Payoff Learning for AVs. *IEEE Trans. VT*, 71(4), 3609-3622.
