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

## 最新实验结果（全 CAV @ 100% 渗透率）

| 模型 | 1200pcu/h | 2000pcu/h | 2800pcu/h | 3600pcu/h | 碰撞 |
|------|:------:|:------:|:------:|:------:|:--:|
| **Game (Ours)** | **65** | **110** | **149** | **190** | 0 |
| No-V2X | 65 | 109 | 147 | 184 | 0 |
| Rule-Based | 40 | 69 | 93 | 117 | 0 |
| SUMO Default | 40 | 13 | 29 | 68 | 2 |

### 渗透率效应（Game 模型）

| 渗透率 | 1200 | 2000 | 2800 | 3600 |
|:------:|:----:|:----:|:----:|:----:|
| 0% | 3 | 4 | 4 | 4 |
| 10% | 3 | 6 | 5 | 5 |
| 30% | 11 | 8 | 7 | 24 |
| 50% | 5 | 4 | 4 | 6 |
| 70% | 2 | 6 | 5 | 4 |
| **100%** | **65** | **110** | **149** | **190** |

> 30%-70% 渗透率存在 "死亡谷"：CAV 不足，博弈失效，效率反而不如全人类。

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
