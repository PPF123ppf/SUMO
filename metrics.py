"""
metrics.py
==========
SUMO-1 项目综合评价指标模块。

在原有效率/安全指标基础上，新增：
  1. 舒适性指标：Jerk 统计（均值、95%分位、超限比例）
  2. 公平性指标：各车延误标准差、延误基尼系数

设计原则：纯函数式，不依赖 traci/SUMO，只做 numpy 计算。
"""

import numpy as np
from typing import List, Dict


# ====================== 舒适性指标（Comfort） ======================

def compute_comfort_metrics(
    acc_history: Dict[str, List[float]],
    step_len: float = 0.1,
    jerk_comfort_threshold: float = 4.0,
) -> Dict[str, float]:
    """从各车加速度历史计算舒适性指标。

    Parameters
    ----------
    acc_history : {vid: [acc_1, acc_2, ...]}，每帧加速度序列 (m/s²)
    step_len   : 仿真步长 (s)
    jerk_comfort_threshold : 舒适性 Jerk 阈值 (m/s³)，超过视为不舒适

    Returns
    -------
    {
        "jerk_mean":      全路段平均 Jerk (m/s³)
        "jerk_p95":       Jerk 95% 分位值
        "jerk_max":       Jerk 最大值
        "jerk_comfort_violation_rate": 超过舒适阈值的 Jerk 样本占比
        "jerk_comfort_violation_cnt":  超过舒适阈值的 Jerk 样本数
    }
    """
    all_jerks = []
    for vid, acc_seq in acc_history.items():
        if len(acc_seq) < 2:
            continue
        acc_arr = np.array(acc_seq, dtype=np.float64)
        jerk = np.abs(np.diff(acc_arr)) / step_len
        all_jerks.extend(jerk.tolist())

    if not all_jerks:
        return {
            "jerk_mean": 0.0,
            "jerk_p95": 0.0,
            "jerk_max": 0.0,
            "jerk_comfort_violation_rate": 0.0,
            "jerk_comfort_violation_cnt": 0,
        }

    jerk_arr = np.array(all_jerks)
    n_total = len(jerk_arr)
    n_viol = int(np.sum(jerk_arr > jerk_comfort_threshold))

    return {
        "jerk_mean": round(float(np.mean(jerk_arr)), 4),
        "jerk_p95": round(float(np.percentile(jerk_arr, 95)), 4),
        "jerk_max": round(float(np.max(jerk_arr)), 4),
        "jerk_comfort_violation_rate": round(n_viol / max(n_total, 1), 4),
        "jerk_comfort_violation_cnt": n_viol,
    }


# ====================== 公平性指标（Fairness） ======================

def gini_coefficient(values: List[float]) -> float:
    """计算基尼系数 = 2 * sum(|x_i - x_j|) / (n * sum(x))。

    取值范围 [0, 1]：
      0 = 完全平等（所有车延误相同）
      1 = 完全不平等（一辆车承担所有延误）
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0 or np.sum(arr) < 1e-12:
        return 0.0
    n = len(arr)
    # 相对均方差的基尼简化公式
    sorted_arr = np.sort(arr)
    cumsum = np.cumsum(sorted_arr)
    numerator = 2.0 * np.sum((np.arange(1, n + 1)) * sorted_arr) - (n + 1) * np.sum(sorted_arr)
    denominator = n * np.sum(sorted_arr)
    # 等价标准形式
    gini = numerator / max(denominator, 1e-12)
    return round(float(gini), 4)


def compute_fairness_metrics(
    vehicle_delays: List[float],
    vehicle_travel_times: List[float],
) -> Dict[str, float]:
    """从各车延误 / 行程时间计算公平性指标。

    Parameters
    ----------
    vehicle_delays       : 各车时间损失（s），长度 = N
    vehicle_travel_times : 各车总行程时间（s），长度 = N

    Returns
    -------
    {
        "delay_std":           延误标准差 (s)
        "delay_cv":            延误变异系数（标准差/均值）
        "delay_gini":          延误基尼系数
        "travel_time_gini":    行程时间基尼系数
        "delay_p90_p10_ratio": 延误 P90/P10 分位比（数值=1 表示完全平等）
    }
    """
    delays = np.array(vehicle_delays, dtype=np.float64)
    ttimes = np.array(vehicle_travel_times, dtype=np.float64)

    if len(delays) == 0:
        return {
            "delay_std": 0.0,
            "delay_cv": 0.0,
            "delay_gini": 0.0,
            "travel_time_gini": 0.0,
            "delay_p90_p10_ratio": 1.0,
        }

    delay_std = float(np.std(delays))
    delay_mean = float(np.mean(delays))
    delay_cv = delay_std / max(abs(delay_mean), 1e-12)

    # 分位比
    p90 = float(np.percentile(delays, 90))
    p10 = float(np.percentile(delays, 10))
    p90_p10 = p90 / max(p10, 1e-12)

    return {
        "delay_std": round(delay_std, 3),
        "delay_cv": round(delay_cv, 4),
        "delay_gini": gini_coefficient(delays.tolist()),
        "travel_time_gini": gini_coefficient(ttimes.tolist()),
        "delay_p90_p10_ratio": round(p90_p10, 3),
    }


# ====================== 综合指标 ======================

def summarize_all_metrics(
    acc_history: Dict[str, List[float]],
    vehicle_delays: List[float],
    vehicle_travel_times: List[float],
    step_len: float = 0.1,
    jerk_comfort_threshold: float = 4.0,
) -> Dict[str, float]:
    """一站式计算所有新增指标，返回平铺 dict 供合并到结果中。"""
    comfort = compute_comfort_metrics(acc_history, step_len, jerk_comfort_threshold)
    fairness = compute_fairness_metrics(vehicle_delays, vehicle_travel_times)
    return {**comfort, **fairness}
