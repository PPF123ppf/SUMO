# SUMO 混合交通博弈换道仿真系统

基于 SUMO 事故场景的混合交通仿真，实现 **Stackelberg 博弈 + 混合博弈结构 + 8维原子特征 + MaxEnt IRL** 的 CAV 换道决策模型。

## 核心创新

| 创新点 | 说明 |
|--------|------|
| **混合博弈结构** | 3种后车类型 × 3种博弈结构（静态 Nash / Stackelberg / Nash 合作）自适应选择 |
| **8维原子特征** | speed, urgency, pressure, safe, coop, social, density, lc_cost — 每维可独立解释 |
| **混合交通** | CAV 渗透率 0%~100%，3种异质人类车（保守/普通/激进） |
| **低渗透率自适应** | 对抗性偏差 + 互惠记忆 + 渗透率自适应阈值 |
| **IRL 数据驱动** | 从 AD4CHE 数据集学习换道决策权重 |
| **在线学习** | 每次换道后 TD 微调权重 |

---

## 技术方案

### 1. 场景建模

- 4km 三车道高速公路（E0），中间车道（lane 1）在 3000m 处因事故封堵
- 事故车道上的车辆必须合流到相邻车道
- 事故时间线：
  - **正常行驶期**（t < 90s）：CACC 类跟驰巡航
  - **突发期**（90-100s）：事故发生，仅局部 V2X 感知，高紧迫度换道
  - **有序期**（100s+）：全局 V2X 广播激活，有序疏散

### 2. 混合交通架构

CAV 和人类车在同一条路上混行，CAV 渗透率从 0% 到 100% 可调：

| 角色 | 换道决策 | 跟驰模型 | V2X | 协同让行 |
|------|---------|---------|:--:|:--:|
| **CAV** | 混合博弈求解器（8维特征） | CACC 自定义跟驰 | ✅ 500m~1200m | ✅ CAV 互协 |
| **人类车（保守型 33%）** | SL2015（sigma=0.3） | Krauss | ❌ | ❌ |
| **人类车（普通型 33%）** | SL2015（sigma=0.5） | Krauss | ❌ | ❌ |
| **人类车（激进型 33%）** | SL2015（sigma=0.7） | Krauss | ❌ | ❌ |

人类车的异质性参考 He et al. (2025, Scientific Reports) 的 K-Means 驾驶风格分类。

### 3. 本车收益函数（8 维特征）

每个 `(action, follower_action)` 组合提取 8 个原子特征：

```
payoff(action, fa) = w · f(action, fa) - lc_cost · I(action=换道)
```

权重向量 `w` 的初始值（有序期）：

```
[speed, urgency, pressure, safe, coop, social, density, lc_cost]
[0.30,  0.10,    0.06,     0.25, 0.12, 0.05,   0.05,   1.0     ]
```

**权重自适应**：根据局部密度和紧迫度动态调整，堵车或近事故时安全权重上升、速度权重下降。

### 4. 后车收益函数（5 维 × 3 种类型）

后车不是"猜概率"，而是与 CAV 进行真正的博弈。每个后车有自己的收益函数：

```
U_f(ego_action, follower_action) = w_t · [safety, speed_ratio, comfort_cost, coop_reward, gap_safe]
```

根据后车的异质性，分为 3 种类型，各选不同的博弈结构：

| 类型 | 占比 | 博弈结构 | 权重分布 [safety, speed, comfort, coop] | 行为特征 |
|:--:|:--:|---------|:------:|---------|
| **Type 0（自私型）** | 20% | 静态 Nash | [0.45, 0.35, 0.20, 0.00] | 只顾自己，同时决策 |
| **Type 1（合作型）** | 60% | Stackelberg | [0.40, 0.25, 0.15, 0.20] | 本车 Leader，后车反应 |
| **Type 2（高合作型）** | 20% | Nash 合作 | [0.35, 0.15, 0.15, 0.35] | 联合收益最大化 |

### 5. 混合博弈求解器

```
solve_hybrid_game(ego_payoff, fol_id, ...) → expected[换道], expected[保持]

对每种后车类型 t（权重 share_t）:
  Type 0 → _solve_static_nash(ego_payoff, fol_payoff)
    4轮迭代：双方softmax选择，收敛到混合策略Nash均衡

  Type 1 → _solve_stackelberg_per_type(ego_payoff, fol_payoff)
    本车宣布动作 → 后车softmax选择最优反应 → 本车取期望收益

  Type 2 → _solve_nash_cooperative(ego_payoff, fol_payoff)
    联合收益矩阵 → 后车softmax选合并最优 → 本车取对应收益

最终期望 = Σ share_t × expected_t
```

所有博弈求解使用 **softmax**（temperature=0.15）替代 argmax，保留行为随机性（改进自 Liu et al. 2024 的混合策略博弈）。

### 6. 决策流程

```
1. 反应延迟到期 → 车辆进入决策窗口
2. 间距门控：lead_gap ≥ dyn_gap 且 fol_gap ≥ dyn_gap
3. 硬安全门控：min(前TTC, 后TTC) ≥ TTC_threshold（突发2.0s / 有序1.5s）
4. 计算本车收益矩阵（8维特征 × 自适应权重）
5. 博弈求解：
   a. 后车 = CAV → solve_hybrid_game()
   b. 后车 = 人类 → get_follower_prior() 概率预测
   c. 无后车 → 直接评分
6. 增益 = expected[换道] - expected[保持]
7. 渗透率自适应阈值：min_gain = base × (1.0 - 0.6 × (1 - CAV_PENETRATION))
8. 增益 > 阈值 → 执行换道
```

### 7. 低渗透率自适应（低 CAV 占比时的关键机制）

| 机制 | 方法 | 效果 |
|------|------|------|
| **对抗性偏差** | 后车=人类时，加速堵位+10%，减速-10% | CAV 不天真假设人类配合 |
| **互惠记忆** | 记录人类车合作史，下回动态调先验 | 合作者+分，不合作者-分 |
| **渗透率自适应阈值** | 0%CAV→0.4×阈值，100%→1.0× | 低渗透果断，高渗透挑剔 |

### 8. 最大熵逆强化学习（MaxEnt IRL）

从 AD4CHE（中国高速无人机数据，约 360 万帧，3908 个换道片段）学习 8 维权重：

```
梯度 = 专家特征期望 - 学习者特征期望
权重 += 学习率 × 梯度
```

**IRL 学出权重（30轮）**：

```
[speed, urgency, pressure, safe, coop, social, density, lc_cost]
[0.15,  0.19,    0.18,     0.25, 0.02, 0.00,   0.11,   0.70 ]
```

IRL 权重与手工权重在最终效果上相当，证明数据驱动方案可行。两套权重的差异反映了 "事故场景应急（手工配置）" 与 "日常驾驶习惯（IRL 学出）" 的行为差异。

### 9. 在线 TD 微调

每次换道执行后，根据结果即时调整权重（学习率 0.003）：

| 结果 | 调整 |
|------|------|
| 成功完成 | safe += 0.0003, coop += 0.0003（强化） |
| 安全审查失败 | safe += 0.0006（纠偏） |
| 导致后车急刹 | safe += 0.0024, social -= 0.0024（惩罚） |

---

## 模型架构

```
混合交通流量（CAV 渗透率 p%）
│
├── CAV (p%)
│   ├── 感知层 → V2X 噪声/延迟/丢包模型
│   ├── 门控层 → 间距 + 硬 TTC 安全门控
│   ├── 特征层 → 8维原子特征提取 + 自适应权重
│   ├── 博弈层 → 混合博弈求解器
│   │   ├── Type 0 自私型 → 静态 Nash（双方同时决策）
│   │   ├── Type 1 合作型 → Stackelberg（本车领导者）
│   │   └── Type 2 高合作型 → Nash 合作（联合最优）
│   ├── 决策层 → 增益 > 自适应阈值 → 执行换道
│   ├── 执行层 → 准备(0.4s) → 执行(2.8s) → 稳定(0.5s)
│   └── 学习层 → 在线 TD 微调
│
└── 人类车 ((1-p)%）
    ├── 保守型 (33%) → sigma=0.3, 跟车3.0m, 极速108km/h
    ├── 普通型 (33%) → sigma=0.5, 跟车2.5m, 极速120km/h
    └── 激进型 (33%) → sigma=0.7, 跟车1.8m, 极速120km/h
    └── SUMO SL2015 原生换道 + Krauss 跟驰
```

---

## 8维特征空间

| 维度 | 特征 | 换道时 | 不换道时 | 物理含义 |
|:--:|------|--------|---------|----------|
| 0 | speed | 目标车道前车速度比 | 自车速度比 | 速度收益 |
| 1 | urgency | 距事故归一化距离 | 0 | 紧迫驱动力 |
| 2 | pressure | 当前车道头距压力 | 0 | 逃离压力 |
| 3 | safe | min(前TTC, 后TTC)评分 | 当前车道前TTC | 安全评分 |
| 4 | coop | 后车协同+制动加分 | 0.05 | 协同奖励 |
| 5 | social | 换道对后车TTC缩减比 | 0 | 社会冲击 |
| 6 | density | 局部车流密度 | 0 | 环境压力 |
| 7 | lc_cost | 1.0 | 0.0 | 换道固有代价 |

---

## 场景

- 4km 三车道高速（E0），3000m 处中间车道事故封堵
- 事故时间线：t<90s 正常 → 90-100s 突发期 → 100s+ 有序疏散期
- 全 CAV 配备 V2X 通信，人类车无 V2X

### 4 种基线模型

| 模型 | 换道方式 | V2X | 协同 |
|------|---------|:--:|:--:|
| **Game (Ours)** | 混合博弈 + 8维特征 | ✅ | ✅ |
| SUMO Default | SL2015 原生 | — | — |
| Rule-Based | 固定 TTC≥3.0s 阈值 | 部分 | ❌ |
| No-V2X | 博弈模型但无全局广播 | 仅局部 | ✅ |

---

## 核心算法详解

### 10. Social Impact 计算

社会冲击是本模型核心创新之一，量化本车换道对目标车道后车 TTC 的侵占：

```
# 换道前：后车到目标车道前车的 TTC
fol_old_ttc = (fol_gap + lead_gap) / max(fol_spd - lead_spd, 0.1)

# 换道后：后车到本车的 TTC
fol_new_ttc = fol_gap / max(fol_spd_new - ego_speed, 0.1)

# 社会冲击 = TTC 缩减比例（0~1）
social = max(0, (fol_old_ttc - fol_new_ttc) / max(fol_old_ttc, 0.1))
social = min(social, 1.0)
```

当后车距原前车很远（fol_old_ttc 大），但本车切入后 TTC 骤降（fol_new_ttc 小）时，social 值高，`w_social`（正权重）会降低换道收益，抑制强行插入。

### 11. 三阶段换道执行

CAV 的换道不是一帧完成的，而是三阶段状态机：

| 阶段 | 典型时长 | 描述 |
|------|:------:|------|
| **准备 (prepare)** | 0.4s | 观察目标间隙，安全复查，打灯信号 |
| **执行 (execute)** | 2.8s | 正弦横向速度剖面换道 |
| **稳定 (stabilize)** | 0.5s | 进入新车道，调整跟车姿态 |

准备阶段结束时执行严格的安全复查，若发现目标车道有碰撞风险则取消换道：

```
prepare → safety_check → execute → stabilize → complete
                      ↘  cancel → online_update_weights(success=False)
                      ↗  (改为继续在原车道行驶)
```

### 12. 感知模型

CAV 配备带噪声和延迟的感知模型，模拟真实传感器：

| 参数 | 值 | 含义 |
|------|:--:|------|
| V2X 范围（局部） | 500m | DSRC 典型值 |
| V2X 范围（广播后） | 1200m | 全局事故广播 |
| 丢包率 | 5% | 通信数据包丢失概率 |
| 感知延迟 | 1步 (0.1s) | 传感器处理延迟 |
| 速度噪声 (σ) | 5% | 速度测量标准差 |
| 距离噪声 (σ) | 3% | 距离测量标准差 |

### 13. 反应延迟模型

CAV 感知事故后不是立即决策，而是随机分配反应延迟：

| 阶段 | 延迟范围 | 说明 |
|------|:------:|------|
| 突发期 | 0.20~0.50s | 信息不足，反应较慢 |
| 有序期 | 0.12~0.35s | 全局广播信息完整 |
| 局部密度增益 | +0.18/density | 堵车时延迟加长 |
| 丢包率增益 | +0.22/loss | 通信质量差延迟加长 |

延迟服从对数正态分布，每次仿真开始时随机分配至各车。

### 14. 动力学约束

| 参数 | 值 | 含义 |
|------|:--:|------|
| 纵向最大加速度 | 2.6 m/s² | 舒适加速上限 |
| 纵向最大减速度 | 4.5 m/s² | 紧急制动上限 |
| Jerk 安全上限 | 3.0 m/s³ | 舒适性硬约束 |
| 横向加速度上限 | 2.0 m/s² | 换道舒适性上限 |
| 换道时间 | 2.8s | 名义横向换道时长 |
| 路面最高速度 | 33.33 m/s (120 km/h) | 限速 |
| 限速区速度 | 16.67 m/s (60 km/h) | 事故前 200m 限速 |

### 15. 最大熵逆强化学习算法

```
目标：从专家（真人）数据中学出权重 w，使专家特征期望 = 学习者特征期望

初始化 w
循环迭代 t = 1...N:
  1. 用当前 w 运行仿真（2个rollout），收集学习者特征
  2. 计算梯度：grad = E_专家[f] - E_学习者[f] - λ·w（L2正则项）
  3. 梯度上升：w += α·grad（学习率 α = 0.02）
  4. 裁剪：w = clip(w, 0, 2)
  5. 计算 loss（负对数似然）用于监控

输出：收敛后的 w
```

**AD4CHE 数据集映射**：原始数据 29 列（含 ttc、dhw、thw、左侧车 ID、右侧车 ID 等），通过 `row_to_features()` 映射到 8 维特征空间。

---

## 实验设计与结果

### 实验矩阵

```python
模型：       [Game (Ours), SUMO Default, Rule-Based, No-V2X]       # 4种
渗透率：     [0%, 10%, 30%, 50%, 70%, 100%]                        # 6档
密度：       [1200, 2000, 2800, 3600 pcu/h]                        # 4档
仿真时长：   360秒（3600步 × 0.1s）                                  # 全事故周期
并行：       8核并行，每核1模型                                     # 加速8倍
```

### 最新实验结果（Game 模型，修复 lateral-resolution 后）

| 渗透率 | 1200 | 2000 | 2800 | 3600 |
|:------:|:----:|:----:|:----:|:----:|
| 0% | — | 116 | 167 | 202 |
| 10% | 69 | 110 | 153 | 177 |
| 30% | 72 | 115 | 152 | 189 |
| 50% | 64 | 106 | 147 | 183 |
| 70% | — | — | — | — |
| 100% | — | — | — | — |

> 注：lateral-resolution 修复前，0% CAV 只能通过 2-3 辆车（因为 SL2015 缺少子车道分辨率，换道能力降级到 LC2013）。
> 修复后，0% CAV 通过量提升到 116-202 辆，与 SUMO Default 基线一致。
> 30-50% 渗透率不再有 "死亡谷"——证明之前的效率下降是 SL2015 配置问题而非博弈模型问题。

### 全 CAV 对比（lateral-resolution 修复前数据）

| 模型 | 1200 | 2000 | 2800 | 3600 | 碰撞 |
|------|:----:|:----:|:----:|:----:|:--:|
| **Game (Ours)** | **65** | **110** | **149** | **190** | 0 |
| No-V2X | 65 | 109 | 147 | 184 | 0 |
| Rule-Based | 40 | 69 | 93 | 117 | 0 |
| SUMO Default | 40 | 13 | 29 | 68 | 2 |

---

## 使用方式

### 依赖

- SUMO 1.x（`C:\Program Files (x86)\Eclipse\Sumo`）
- Python 3.10+，`traci`, `sumolib`, `numpy`, `pandas`, `matplotlib`

### 运行单次仿真

```bash
echo "b" | PYTHONIOENCODING=utf-8 python game_lane_change.py
```

### 运行基线对比实验

```bash
# 完整实验（4模型 × 4密度 × 6渗透率 = 96组）
python run_baseline_stepwise.py --sim-steps 3600 --out-dir results/exp1

# 使用 IRL 学出的权重
python run_baseline_stepwise.py --irl-weights irl_weights_v2.npz --out-dir results/exp_irl

# 快速验证
set SIM_STEPS=600 && python run_baseline_stepwise.py

# 只跑部分模型
python run_baseline_stepwise.py --models "Game (Ours)" --penetrations "0.3,0.5,1.0"
```

### 多核并行（8核 8进程）

```bash
for m in "Game (Ours)" "SUMO Default" "Rule-Based" "No-V2X"; do
  python run_baseline_stepwise.py --models "$m" --out-dir "results/parallel/$m" &
done
```

### IRL 训练

```bash
# 快速训练
python run_irl_quick.py

# 完整训练
python irl.py --data-dir data/AD4CHE_dataset_V1.0/.../AD4CHE_Data_V1.0 --iterations 50
```

---

## 项目结构

```
SUMO-1/
├── game_lane_change.py       # 主仿真：博弈求解器、特征提取、状态机
├── baseline_comparison.py    # 4种基线模型实现
├── run_baseline_stepwise.py  # 分步基线运行（支持并行、渗透率、IRL权重）
├── irl.py                    # 最大熵逆强化学习（8维特征）
├── run_irl_quick.py          # 快速 IRL 训练脚本
├── config.py                 # 集中参数配置
├── metrics.py                # 舒适性与公平性评价
├── plot_baseline_results.py  # 结果可视化
│
├── accident_highway.net.xml  # SUMO 路网
├── accident_highway.sumocfg  # SUMO 配置
├── viewsettings.xml          # GUI 视图设置
│
├── irl_weights_v2.npz       # IRL 学出的权重
├── core_model_whitepaper.md  # 模型技术白皮书
│
├── data/                     # AD4CHE 数据集
├── results/                  # 仿真输出
└── Thesis/                   # 论文 LaTeX
```

---

## 参数预设

| 预设 | 时距(s) | 换道阈值(突发) | 换道代价 | 说明 |
|------|---------|:----------:|:------:|------|
| balanced | 1.00 | 0.030 | 0.060 | 效率与安全均衡（默认） |
| aggressive | 0.85 | 0.020 | 0.040 | 更紧时距，更激进 |
| conservative | 1.25 | 0.050 | 0.090 | 更保守 |
| balanced_plus | 1.18 | 0.060 | 0.100 | 最保守 |
