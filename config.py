"""
配置文件 —— 全 CAV 事故场景博弈仿真参数

本文件从 game_lane_change.py 中提取所有可配置参数，
不依赖 traci / sumolib 等 SUMO 运行时模块，仅使用标准库与 numpy。
"""

import math
import numpy as np

# ====================== 仿真基础参数 ======================
STEP_LEN = 0.1          # 仿真步长 (s)
SIM_STEPS_DEFAULT = 3600  # 默认仿真总步数

# ====================== 事故参数 ======================
ACCIDENT_START       = 3000.0   # 事故区起点 (m)
ACCIDENT_END         = 3200.0   # 事故区终点 (m)
ACCIDENT_LANE        = 1        # 封锁车道（中间车道，迫使双向合流）
ACCIDENT_TIME        = 90.0     # s，仿真开始后多久触发事故
ACCIDENT_SEARCH_WINDOW = 120.0  # m，在 ACCIDENT_START 范围内寻找候选事故车辆
BROADCAST_DELAY      = 10.0     # s，事故发生后全局 V2X 广播激活延迟
SLOW_ZONE_START      = 2800.0   # m，限速预警区起点（事故前 200m）
SLOW_SPEED           = 16.67    # m/s，限速区速度上限 (60 km/h)
OBSTACLE_IDS: list = []         # 障碍物车辆 ID 列表（运行时填充）

# ====================== 通信参数 ======================
V2X_RANGE            = 500.0    # m，CAV 常规 V2X 通信范围 (DSRC 典型值)
GLOBAL_V2X_RANGE     = 1200.0   # m，全局事故广播范围
PACKET_LOSS_RATE     = 0.05     # 通信丢包率 (<= 5%), 仅 V2X_CHANNEL="ideal" 时生效
PERC_DELAY_STEPS     = 1        # 感知延迟步数 (1 步 = 0.1 s = 100 ms), 仅 ideal 时生效
V2X_CHANNEL          = "ideal"  # 信道模型: "ideal" | "realistic"
SPEED_NOISE_STD      = 0.05     # 速度感知噪声标准差 (±5%, 1σ)
DIST_NOISE_STD       = 0.03     # 距离感知噪声标准差 (±3%, 1σ)

# ====================== 反应延迟参数 ======================
REACTION_DELAY_CAV_SUDDEN   = (0.20, 0.50)  # s，突发期反应延迟范围
REACTION_DELAY_CAV_INFORMED = (0.12, 0.35)  # s，有序期反应延迟范围
REACTION_DELAY_MIN      = 0.12   # s，反应延迟下限
REACTION_DELAY_MAX      = 0.80   # s，反应延迟上限
REACTION_DELAY_LOGLN_SIGMA = 0.30      # 对数正态分布离散度
REACTION_DELAY_DENSITY_GAIN = 0.18     # 局部密度对反应延迟增益
REACTION_DELAY_LOSS_GAIN    = 0.22     # 丢包率对反应延迟增益

# ====================== 安全参数 ======================
MIN_SAFE_GAP         = 3.0      # m，最小安全间距
TTC_HARD_MIN_SUDDEN  = 2.00     # s，突发期硬安全 TTC 阈值
TTC_HARD_MIN_INFORMED = 1.50    # s，有序期硬安全 TTC 阈值

# ====================== 紧急制动参数 ======================
EMERGENCY_REACT_TIME     = 0.60   # s，障碍前紧急制动覆盖的总反应时间
EMERGENCY_EFF_DEC        = 4.00   # m/s²，紧急制动可用有效减速度
EMERGENCY_MARGIN         = 6.0    # m，停车安全冗余
EMERGENCY_TRIGGER_BUFFER = 5.0    # m，进入覆盖的触发缓冲
EMERGENCY_MAX_ZONE       = 240.0  # m，仅在障碍前该范围内启用覆盖
EMERGENCY_PREP_ABORT_DIST = 70.0  # m，近障碍时取消准备态换道
EMERGENCY_FORCE_BRAKE_DIST = 95.0 # m，近障碍强制进入紧急制动覆盖
EMERGENCY_NO_LC_DIST     = 90.0   # m，近障碍绝对禁止发起新换道

# ====================== 换道博弈参数 ======================
GAME_MIN_GAIN_SUDDEN   = 0.025   # 突发期换道收益最小增益阈值
GAME_MIN_GAIN_INFORMED = 0.008   # 有序期换道收益最小增益阈值
LANE_CHANGE_COST       = 0.05    # 换道动作固定代价（舒适性/执行风险）
LC_COOLDOWN            = 1.5     # s，单车换道冷却时间
PREP_ADMIT_LIMIT_SUDDEN   = 1    # 每步突发期最多新增 prepare 车辆数
PREP_ADMIT_LIMIT_INFORMED = 2    # 每步有序期最多新增 prepare 车辆数
MAX_QUEUE_ALLOWED         = 3    # 队列协调时同时允许换道的最大车辆数

LOCAL_DENSITY_RANGE     = 70.0   # m，局部密度统计范围
LOCAL_DENSITY_NORM      = 10.0   # 局部密度归一化参考车辆数

# ====================== 协同换道参数 ======================
PLATOON_HEADWAY   = 15.0    # m，判定同编队/同协同簇的头距阈值
COOP_MIN_GAP      = 8.0     # m，协同让行希望形成的最小目标间隙
COOP_REQUEST_GAP  = 25.0    # m，小于该值时触发协同让行请求
COOP_SLOW_DELTA   = 2.5     # m/s，目标车道后车为让行而减速的幅度
COOP_RECOVER_DIST = 60.0    # m，完成协同后恢复控制的纵向距离

# ====================== 动力学约束 ======================
LANE_WIDTH          = 3.2     # m，SUMO 默认车道宽度
MAX_LAT_ACC         = 2.0     # m/s²，横向加速度上限
# 正弦横向速度剖面峰值横向加速度公式: a_lat_peak = pi^2 * D / (2 * T^2)
# D = 3.2 m, a_lat <= 2.0 m/s^2 -> T_min = pi * sqrt(D / (2 * a)) ≈ 2.81 s
MIN_LC_EXEC_DURATION = float(math.pi * math.sqrt(LANE_WIDTH / (2.0 * MAX_LAT_ACC)))
LC_DURATION_CAV     = 2.8     # s，CAV 执行阶段名义换道时长
LC_PREP_CAV         = 0.4     # s，准备阶段（观察与预判）
LC_STAB_CAV         = 0.5     # s，稳定阶段（完成后姿态恢复）

# ====================== 常态巡航参数 ======================
NORMAL_HEADWAY          = 1.0    # s，常态巡航目标时距
NORMAL_GAP_GAIN         = 0.22   # 间距误差反馈增益
NORMAL_REL_SPEED_GAIN   = 0.40   # 相对速度反馈增益
NORMAL_FREEFLOW_GAIN    = 0.12   # 无前车时速度回归增益

# ====================== 仿真动力学约束 ======================
MAX_LONG_ACC      = 2.6     # m/s²，纵向最大加速度（仿真控制器约束）
MAX_LONG_DEC      = 4.5     # m/s²，纵向最大减速度（仿真控制器约束）
MAX_JERK_LIMIT    = 3.0     # m/s³，舒适性硬约束：最大跃度上限（仿真监控阈值）
FOLLOWER_BEHAV_WINDOW = 15  # 步，后车行为在线估计窗口

# ====================== Level-k 认知层级参数 ======================
LEVEL_K_MAX      = 2                      # 最高认知层级
LEVEL_K_DIST     = [0.20, 0.60, 0.20]     # Level-0/1/2 分布概率

ACTIVE_PROFILE         = "balanced"

# ====================== 舒适性 & 公平性评价参数（独立于仿真约束） ======================
JERK_COMFORT_THRESHOLD = 4.0     # m/s³，超过此值视为不舒适（舒适范围通常 <2.0~2.5 m/s³, 容忍上限 4.0）
MAX_LONG_ACC_METRIC    = 3.0     # m/s²，用于评价的纵向加速度上限
MAX_LONG_DEC_METRIC    = 5.0     # m/s²，用于评价的纵向减速度上限
MAX_JERK_METRIC_LIMIT  = 10.0    # m/s³，超出视为 jerk 违例（动力学约束违规，非舒适性评价）
FOLLOWER_BEHAV_WINDOW_METRIC = 20  # 步数，后车行为历史窗口（评价用）

# ====================== 参数标定预设 ======================
PROFILE_PRESETS = {
    "balanced": {
        "normal_headway": 1.00,
        "normal_gap_gain": 0.22,
        "normal_rel_speed_gain": 0.40,
        "normal_freeflow_gain": 0.12,
        "game_min_gain_sudden": 0.030,
        "game_min_gain_informed": 0.010,
        "lane_change_cost": 0.060,
        "coop_min_gap": 12.0,
        "coop_request_gap": 30.0,
        "coop_slow_delta": 3.0,
    },
    "balanced_plus": {
        "normal_headway": 1.18,
        "normal_gap_gain": 0.18,
        "normal_rel_speed_gain": 0.34,
        "normal_freeflow_gain": 0.08,
        "game_min_gain_sudden": 0.060,
        "game_min_gain_informed": 0.025,
        "lane_change_cost": 0.100,
        "coop_min_gap": 15.0,
        "coop_request_gap": 36.0,
        "coop_slow_delta": 3.2,
    },
    "conservative": {
        "normal_headway": 1.25,
        "normal_gap_gain": 0.18,
        "normal_rel_speed_gain": 0.34,
        "normal_freeflow_gain": 0.08,
        "game_min_gain_sudden": 0.050,
        "game_min_gain_informed": 0.025,
        "lane_change_cost": 0.090,
        "coop_min_gap": 14.0,
        "coop_request_gap": 34.0,
        "coop_slow_delta": 2.5,
    },
    "aggressive": {
        "normal_headway": 0.85,
        "normal_gap_gain": 0.28,
        "normal_rel_speed_gain": 0.50,
        "normal_freeflow_gain": 0.16,
        "game_min_gain_sudden": 0.020,
        "game_min_gain_informed": 0.005,
        "lane_change_cost": 0.040,
        "coop_min_gap": 10.0,
        "coop_request_gap": 26.0,
        "coop_slow_delta": 3.4,
    },
}


# ====================== 辅助函数 ======================

def apply_parameter_profile(profile_name: str = "balanced") -> str:
    """
    应用参数预设，返回实际生效的预设名。

    修改本模块的全局变量 NORMAL_HEADWAY, NORMAL_GAP_GAIN, NORMAL_REL_SPEED_GAIN,
    NORMAL_FREEFLOW_GAIN, GAME_MIN_GAIN_SUDDEN, GAME_MIN_GAIN_INFORMED,
    LANE_CHANGE_COST, COOP_MIN_GAP, COOP_REQUEST_GAP, COOP_SLOW_DELTA 以及 ACTIVE_PROFILE。
    """
    import sys as _sys

    # 通过本模块的全局命名空间进行修改
    _mod = _sys.modules[__name__]

    aliases = {
        "b": "balanced",
        "bp": "balanced_plus",
        "p": "balanced_plus",
        "balanced+": "balanced_plus",
        "c": "conservative",
        "a": "aggressive",
    }
    key_raw = str(profile_name or "").strip().lower()
    key = aliases.get(key_raw, key_raw if key_raw else "balanced")
    if key not in PROFILE_PRESETS:
        key = "balanced"

    cfg = PROFILE_PRESETS[key]
    _mod.NORMAL_HEADWAY = cfg["normal_headway"]
    _mod.NORMAL_GAP_GAIN = cfg["normal_gap_gain"]
    _mod.NORMAL_REL_SPEED_GAIN = cfg["normal_rel_speed_gain"]
    _mod.NORMAL_FREEFLOW_GAIN = cfg["normal_freeflow_gain"]
    _mod.GAME_MIN_GAIN_SUDDEN = cfg["game_min_gain_sudden"]
    _mod.GAME_MIN_GAIN_INFORMED = cfg["game_min_gain_informed"]
    _mod.LANE_CHANGE_COST = cfg["lane_change_cost"]
    _mod.COOP_MIN_GAP = cfg["coop_min_gap"]
    _mod.COOP_REQUEST_GAP = cfg["coop_request_gap"]
    _mod.COOP_SLOW_DELTA = cfg["coop_slow_delta"]
    _mod.ACTIVE_PROFILE = key
    return key


def get_config() -> dict:
    """
    返回当前所有配置参数的字典（用于日志记录/报告）。

    每次调用时从模块当前全局变量中读取，因此可反映 apply_parameter_profile()
    对可调参数的修改。
    """
    import sys as _sys
    _mod = _sys.modules[__name__]
    _g = vars(_mod)

    return {
        # 仿真基础
        "STEP_LEN": _g["STEP_LEN"],
        "SIM_STEPS_DEFAULT": _g["SIM_STEPS_DEFAULT"],

        # 事故参数
        "ACCIDENT_START": _g["ACCIDENT_START"],
        "ACCIDENT_END": _g["ACCIDENT_END"],
        "ACCIDENT_LANE": _g["ACCIDENT_LANE"],
        "ACCIDENT_TIME": _g["ACCIDENT_TIME"],
        "ACCIDENT_SEARCH_WINDOW": _g["ACCIDENT_SEARCH_WINDOW"],
        "BROADCAST_DELAY": _g["BROADCAST_DELAY"],
        "SLOW_ZONE_START": _g["SLOW_ZONE_START"],
        "SLOW_SPEED": _g["SLOW_SPEED"],

        # 通信参数
        "V2X_RANGE": _g["V2X_RANGE"],
        "GLOBAL_V2X_RANGE": _g["GLOBAL_V2X_RANGE"],
        "PACKET_LOSS_RATE": _g["PACKET_LOSS_RATE"],
        "PERC_DELAY_STEPS": _g["PERC_DELAY_STEPS"],
        "V2X_CHANNEL": _g["V2X_CHANNEL"],
        "SPEED_NOISE_STD": _g["SPEED_NOISE_STD"],
        "DIST_NOISE_STD": _g["DIST_NOISE_STD"],

        # 反应延迟
        "REACTION_DELAY_CAV_SUDDEN": _g["REACTION_DELAY_CAV_SUDDEN"],
        "REACTION_DELAY_CAV_INFORMED": _g["REACTION_DELAY_CAV_INFORMED"],
        "REACTION_DELAY_MIN": _g["REACTION_DELAY_MIN"],
        "REACTION_DELAY_MAX": _g["REACTION_DELAY_MAX"],
        "REACTION_DELAY_LOGLN_SIGMA": _g["REACTION_DELAY_LOGLN_SIGMA"],
        "REACTION_DELAY_DENSITY_GAIN": _g["REACTION_DELAY_DENSITY_GAIN"],
        "REACTION_DELAY_LOSS_GAIN": _g["REACTION_DELAY_LOSS_GAIN"],

        # 安全参数
        "MIN_SAFE_GAP": _g["MIN_SAFE_GAP"],
        "TTC_HARD_MIN_SUDDEN": _g["TTC_HARD_MIN_SUDDEN"],
        "TTC_HARD_MIN_INFORMED": _g["TTC_HARD_MIN_INFORMED"],

        # 紧急制动
        "EMERGENCY_REACT_TIME": _g["EMERGENCY_REACT_TIME"],
        "EMERGENCY_EFF_DEC": _g["EMERGENCY_EFF_DEC"],
        "EMERGENCY_MARGIN": _g["EMERGENCY_MARGIN"],
        "EMERGENCY_TRIGGER_BUFFER": _g["EMERGENCY_TRIGGER_BUFFER"],
        "EMERGENCY_MAX_ZONE": _g["EMERGENCY_MAX_ZONE"],
        "EMERGENCY_PREP_ABORT_DIST": _g["EMERGENCY_PREP_ABORT_DIST"],
        "EMERGENCY_FORCE_BRAKE_DIST": _g["EMERGENCY_FORCE_BRAKE_DIST"],
        "EMERGENCY_NO_LC_DIST": _g["EMERGENCY_NO_LC_DIST"],

        # 换道博弈
        "GAME_MIN_GAIN_SUDDEN": _g["GAME_MIN_GAIN_SUDDEN"],
        "GAME_MIN_GAIN_INFORMED": _g["GAME_MIN_GAIN_INFORMED"],
        "LANE_CHANGE_COST": _g["LANE_CHANGE_COST"],
        "LC_COOLDOWN": _g["LC_COOLDOWN"],
        "PREP_ADMIT_LIMIT_SUDDEN": _g["PREP_ADMIT_LIMIT_SUDDEN"],
        "PREP_ADMIT_LIMIT_INFORMED": _g["PREP_ADMIT_LIMIT_INFORMED"],

        "LOCAL_DENSITY_RANGE": _g["LOCAL_DENSITY_RANGE"],
        "LOCAL_DENSITY_NORM": _g["LOCAL_DENSITY_NORM"],

        # 协同换道
        "PLATOON_HEADWAY": _g["PLATOON_HEADWAY"],
        "COOP_MIN_GAP": _g["COOP_MIN_GAP"],
        "COOP_REQUEST_GAP": _g["COOP_REQUEST_GAP"],
        "COOP_SLOW_DELTA": _g["COOP_SLOW_DELTA"],
        "COOP_RECOVER_DIST": _g["COOP_RECOVER_DIST"],

        # 动力学约束
        "LANE_WIDTH": _g["LANE_WIDTH"],
        "MAX_LAT_ACC": _g["MAX_LAT_ACC"],
        "MIN_LC_EXEC_DURATION": _g["MIN_LC_EXEC_DURATION"],
        "LC_DURATION_CAV": _g["LC_DURATION_CAV"],
        "LC_PREP_CAV": _g["LC_PREP_CAV"],
        "LC_STAB_CAV": _g["LC_STAB_CAV"],

        # 常态巡航
        "NORMAL_HEADWAY": _g["NORMAL_HEADWAY"],
        "NORMAL_GAP_GAIN": _g["NORMAL_GAP_GAIN"],
        "NORMAL_REL_SPEED_GAIN": _g["NORMAL_REL_SPEED_GAIN"],
        "NORMAL_FREEFLOW_GAIN": _g["NORMAL_FREEFLOW_GAIN"],

        # 仿真动力学约束
        "MAX_LONG_ACC": _g["MAX_LONG_ACC"],
        "MAX_LONG_DEC": _g["MAX_LONG_DEC"],
        "MAX_JERK_LIMIT": _g["MAX_JERK_LIMIT"],
        "FOLLOWER_BEHAV_WINDOW": _g["FOLLOWER_BEHAV_WINDOW"],

        # Level-k 认知层级
        "LEVEL_K_MAX": _g["LEVEL_K_MAX"],
        "LEVEL_K_DIST": _g["LEVEL_K_DIST"],

        "ACTIVE_PROFILE": _g["ACTIVE_PROFILE"],

        # 舒适性 & 公平性评价参数
        "JERK_COMFORT_THRESHOLD": _g["JERK_COMFORT_THRESHOLD"],
        "MAX_LONG_ACC_METRIC": _g["MAX_LONG_ACC_METRIC"],
        "MAX_LONG_DEC_METRIC": _g["MAX_LONG_DEC_METRIC"],
        "MAX_JERK_METRIC_LIMIT": _g["MAX_JERK_METRIC_LIMIT"],
        "FOLLOWER_BEHAV_WINDOW_METRIC": _g["FOLLOWER_BEHAV_WINDOW_METRIC"],
    }
