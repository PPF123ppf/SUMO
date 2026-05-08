import os
import sys
import shutil
import traci
import sumolib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from datetime import datetime
import metrics as metrics_mod  # 综合评价指标模块

# ——— SUMO 环境 ———
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
sumo_bin = r'C:\Program Files (x86)\Eclipse\Sumo\bin'
if sumo_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = sumo_bin + os.pathsep + os.environ.get('PATH', '')

# ====================== 全局参数 ======================
ACCIDENT_START  = 3000.0   # 事故区起点(m)
ACCIDENT_END    = 3200.0   # 事故区终点(m)
ACCIDENT_LANE   = 1        # 封锁车道（中间车道，迫使双向合流）
V2X_RANGE       = 500.0    # CAV通信范围(m) — DSRC典型值
GLOBAL_V2X_RANGE = 1200.0  # 全局事故广播范围(m)
MIN_SAFE_GAP    = 3.0      # 最小安全间距(m)
SLOW_ZONE_START = 2800.0   # 限速预警区起点（事故前200m）
SLOW_SPEED      = 16.67    # 限速区速度上限 60km/h
OBSTACLE_IDS: list = []
STEP_LEN         = 0.1      # 仿真步长(s)
PLATOON_HEADWAY  = 15.0    # m，判定同编队/同协同簇的头距阈值
COOP_MIN_GAP     = 8.0     # m，协同让行希望形成的最小目标间隙（CAV 精度更高，更小间隙即可）
COOP_REQUEST_GAP = 25.0    # m，小于该值时触发协同让行请求
COOP_SLOW_DELTA  = 2.5     # m/s，目标车道后车为让行而减速的幅度（更柔和）
COOP_RECOVER_DIST = 60.0   # m，完成协同后恢复控制的纵向距离

# ====================== 事故阶段参数 ======================
ACCIDENT_TIME           = 90.0   # s，仿真开始后多久触发事故（前段为正常行驶期）
ACCIDENT_SEARCH_WINDOW  = 120.0  # m，在 ACCIDENT_START ± 此范围内寻找候选事故车辆
BROADCAST_DELAY         = 10.0   # s，事故发生后全局V2X广播激活延迟
REACTION_DELAY_CAV_SUDDEN   = (0.20, 0.50)  # s，突发期反应延迟范围（CAV 实际 0.1~0.3s，考虑感知+决策额外开销）
REACTION_DELAY_CAV_INFORMED = (0.12, 0.35)  # s，有序期反应延迟范围（有全局广播信息，反应更快）
REACTION_DELAY_MIN      = 0.12  # s，反应延迟下限（与 CAV 有序期下限匹配）
REACTION_DELAY_MAX      = 0.80  # s，反应延迟上限（CAV 极限不超过0.8s）
REACTION_DELAY_LOGLN_SIGMA = 0.30   # 对数正态分布离散度（更紧凑）
REACTION_DELAY_DENSITY_GAIN = 0.18  # 局部密度对反应延迟增益
REACTION_DELAY_LOSS_GAIN    = 0.22  # 丢包率对反应延迟增益
GAME_MIN_GAIN_SUDDEN    = 0.025  # 突发期换道收益最小增益阈值
GAME_MIN_GAIN_INFORMED  = 0.008  # 有序期换道收益最小增益阈值
LANE_CHANGE_COST        = 0.05   # 换道动作固定代价（舒适性/执行风险）
TTC_HARD_MIN_SUDDEN     = 2.00   # s，突发期硬安全TTC阈值（CAV 比人类快，1.5~2.0s 足矣）
TTC_HARD_MIN_INFORMED   = 1.50   # s，有序期硬安全TTC阈值（有全局信息更精确）
LC_COOLDOWN             = 1.5    # s，单车换道冷却时间（CAV 反应快，不需要等3秒）
LOCAL_DENSITY_RANGE     = 70.0   # m，局部密度统计范围
LOCAL_DENSITY_NORM      = 10.0   # 局部密度归一化参考车辆数
EMERGENCY_REACT_TIME    = 0.60   # s，障碍前紧急制动覆盖的总反应时间
EMERGENCY_EFF_DEC       = 4.00   # m/s²，紧急制动可用有效减速度（保守值）
EMERGENCY_MARGIN        = 6.0    # m，停车安全冗余
EMERGENCY_TRIGGER_BUFFER = 5.0   # m，进入覆盖的触发缓冲
EMERGENCY_MAX_ZONE      = 240.0  # m，仅在障碍前该范围内启用覆盖
EMERGENCY_PREP_ABORT_DIST = 70.0 # m，近障碍时取消准备态换道，避免来不及并线
EMERGENCY_FORCE_BRAKE_DIST = 95.0 # m，近障碍强制进入紧急制动覆盖
EMERGENCY_NO_LC_DIST    = 90.0   # m，近障碍绝对禁止发起新换道
PREP_ADMIT_LIMIT_SUDDEN = 1      # 每步突发期最多新增prepare车辆数
PREP_ADMIT_LIMIT_INFORMED = 2    # 每步有序期最多新增prepare车辆数

# 事故阶段状态（每次 run_once 开始前重置）
_accident_state: dict = {
    "happened":         False,   # 事故是否已发生
    "time_actual":      -1.0,    # 事故实际发生时刻(s)
    "broadcast_active": False,   # 全局V2X广播是否已激活
}
_reaction_delays: dict = {}      # {vid: aware_deadline_step}，各车反应到期步数
_aware_start_steps: dict = {}    # {vid: aware_start_step}，各车首次感知事故时刻
_phase_aware_vids: set = set()   # 已分配反应延迟的车辆集合
_last_lc_step: dict = {}         # {vid: last_lc_step}，每辆车最近换道步数

# ====================== 动力学约束 ======================
LANE_WIDTH       = 3.2    # m，SUMO 默认车道宽度
# 正弦横向速度剖面峰值横向加速度公式：a_lat_peak = π²·D / (2·T²)
# D=3.2m, a_lat≤2.0m/s² → T_min = π·sqrt(D/(2·a)) ≈ 2.81s
MAX_LAT_ACC      = 2.0    # m/s²，横向加速度上限
MIN_LC_EXEC_DURATION = float(np.pi * np.sqrt(LANE_WIDTH / (2.0 * MAX_LAT_ACC)))
LC_DURATION_CAV  = 2.8    # s，CAV 执行阶段名义换道时长
LC_PREP_CAV      = 0.4    # s，准备阶段（观察与预判）
LC_STAB_CAV      = 0.5    # s，稳定阶段（完成后姿态恢复）
_lc_state: dict = {}      # {vid: {phase, from_lane, target_lane, prep_end, exec_end, stab_end, exec_dur, stab_dur}}

# 仿真性能优化参数
DECISION_INTERVAL = 3   # 每隔 N 步做一次博弈决策（0.3s 间隔，对换道时序影响可忽略）
_max_speed_cache: dict = {}  # {type_id: max_speed} 速度缓存

def cached_max_speed(vid: str) -> float:
    """带缓存的最大速度查询（同类型车速度相同，避免重复 TraCI 调用）。"""
    try:
        vtype = traci.vehicle.getTypeID(vid)
    except traci.exceptions.TraCIException:
        return 33.33
    if vtype not in _max_speed_cache:
        try:
            _max_speed_cache[vtype] = traci.vehicle.getMaxSpeed(vid)
        except traci.exceptions.TraCIException:
            _max_speed_cache[vtype] = 33.33
    return _max_speed_cache[vtype]

# ====================== 全CAV常态巡航参数 ======================
NORMAL_HEADWAY          = 1.0    # s，常态巡航目标时距
NORMAL_GAP_GAIN         = 0.22   # 间距误差反馈增益
NORMAL_REL_SPEED_GAIN   = 0.40   # 相对速度反馈增益
NORMAL_FREEFLOW_GAIN    = 0.12   # 无前车时速度回归增益

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
ACTIVE_PROFILE = "balanced"

# ====================== 混合交通参数 ======================
CAV_PENETRATION    = 0.30   # CAV 渗透率 (0.0~1.0)
HUMAN_SIGMA        = 0.50   # 人类车随机性
HUMAN_MIN_GAP      = 2.50   # m，人类车最小跟车间距

def is_vehicle_cav(vid: str) -> bool:
    """判断车辆是否为 CAV（基于车辆类型 ID）。"""
    try:
        return traci.vehicle.getTypeID(vid) == "cav"
    except traci.exceptions.TraCIException:
        return False

# 互惠记忆（改进 #3）：记录人类车的合作行为
# {vid: cooperation_score}, 0.0=不合作, 1.0=很合作, 0.5=初始未知
_human_coop_memory: dict = {}

# ====================== Level-k 认知层级参数 ======================
LEVEL_K_MAX      = 2                      # 最高认知层级
LEVEL_K_DIST     = [0.20, 0.60, 0.20]     # Level-0/1/2 分布概率

# ====================== 感知模型 ======================
PERC_DELAY_STEPS = 1      # 感知延迟步数（1步 = 0.1s = 100ms）
SPEED_NOISE_STD  = 0.05   # 速度感知噪声标准差（±5%，1σ）
DIST_NOISE_STD   = 0.03   # 距离感知噪声标准差（±3%，1σ）
PACKET_LOSS_RATE = 0.05   # 通信丢包率（≤5%）
MAX_JERK_LIMIT   = 3.0    # 舒适性硬约束：最大跃度(Jerk)上限 m/s³
MAX_LONG_ACC     = 2.6    # 纵向最大加速度 m/s²
MAX_LONG_DEC     = 4.5    # 纵向最大减速度 m/s²
FOLLOWER_BEHAV_WINDOW = 15   # 步，后车行为在线估计窗口

_perc_buf: dict  = {}     # {vid: [(step, lead_id, lg, fol_id, fg), ...]}，感知历史缓冲区
_cur_step: int   = 0      # 当前仿真步（由 run_once 主循环更新）
_coop_state: dict = {}    # {supporter_vid: {ego_vid, request_step, init_speed, responded, gap_ready, success}}
_acc_buf: dict   = {}     # 记录上一帧加速度 {vid: acc}
_fol_acc_hist: dict = {}  # {vid: [acc,...]}，后车近期加速度轨迹
_perf_metrics    = {"packet_loss_cnt": 0, "comm_msgs": 0, "jerk_violations": 0, "acc_violations": 0, "energy_g": 0.0}
_normal_ctrl_vids: set = set()   # 正在被常态巡航控制的车辆
_emergency_brake_vids: set = set()  # 当前受紧急制动覆盖的车辆

# ====================== 辅助函数 ======================
def apply_parameter_profile(profile_name: str = "balanced") -> str:
    """应用参数预设，返回实际生效的预设名。"""
    global ACTIVE_PROFILE
    global NORMAL_HEADWAY, NORMAL_GAP_GAIN, NORMAL_REL_SPEED_GAIN, NORMAL_FREEFLOW_GAIN
    global GAME_MIN_GAIN_SUDDEN, GAME_MIN_GAIN_INFORMED, LANE_CHANGE_COST
    global COOP_MIN_GAP, COOP_REQUEST_GAP, COOP_SLOW_DELTA

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
    NORMAL_HEADWAY = cfg["normal_headway"]
    NORMAL_GAP_GAIN = cfg["normal_gap_gain"]
    NORMAL_REL_SPEED_GAIN = cfg["normal_rel_speed_gain"]
    NORMAL_FREEFLOW_GAIN = cfg["normal_freeflow_gain"]
    GAME_MIN_GAIN_SUDDEN = cfg["game_min_gain_sudden"]
    GAME_MIN_GAIN_INFORMED = cfg["game_min_gain_informed"]
    LANE_CHANGE_COST = cfg["lane_change_cost"]
    COOP_MIN_GAP = cfg["coop_min_gap"]
    COOP_REQUEST_GAP = cfg["coop_request_gap"]
    COOP_SLOW_DELTA = cfg["coop_slow_delta"]
    ACTIVE_PROFILE = key
    return key

def gen_rou_xml(n_total: int, path: str = "tmp_routes.rou.xml"):
    """按混合交通动态生成路由文件（CAV + 3种异质人类车）。"""
    n_cav = int(n_total * CAV_PENETRATION)
    n_human = n_total - n_cav
    n_h_each = n_human // 3
    n_h_rem = n_human - n_h_each * 3
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="cav" accel="2.6" decel="4.5" sigma="0.0" length="4.5" minGap="1.5" maxSpeed="33.33" guiShape="passenger" color="0,200,0" lcKeepRight="0" lcSpeedGain="0" lcStrategic="0" lcCooperative="0"/>
    <vType id="h_cons" accel="1.8" decel="4.5" sigma="0.3" length="4.5" minGap="3.0" maxSpeed="30.0" guiShape="passenger" color="100,150,255" laneChangeModel="SL2015" lcStrategic="0.8" lcCooperative="1.0" lcSpeedGain="0.6" lcKeepRight="1.0" lcAssertive="0.2"/>
    <vType id="h_norm" accel="2.0" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="33.33" guiShape="passenger" color="255,200,0" laneChangeModel="SL2015" lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0" lcKeepRight="1.0" lcAssertive="0.5"/>
    <vType id="h_aggr" accel="2.4" decel="4.5" sigma="0.7" length="4.5" minGap="1.8" maxSpeed="33.33" guiShape="passenger" color="255,100,50" laneChangeModel="SL2015" lcStrategic="1.0" lcCooperative="0.6" lcSpeedGain="1.0" lcKeepRight="0.5" lcAssertive="0.8"/>
    <route id="r0" edges="E0"/>
    <flow id="f_cav" type="cav" route="r0" begin="0" end="360" number="{n_cav}" departSpeed="max" departLane="random"/>
    <flow id="f_cons" type="h_cons" route="r0" begin="0" end="360" number="{n_h_each + (1 if n_h_rem > 0 else 0)}" departSpeed="max" departLane="random"/>
    <flow id="f_norm" type="h_norm" route="r0" begin="0" end="360" number="{n_h_each + (1 if n_h_rem > 1 else 0)}" departSpeed="max" departLane="random"/>
    <flow id="f_aggr" type="h_aggr" route="r0" begin="0" end="360" number="{n_h_each}" departSpeed="max" departLane="random"/>
</routes>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path

def get_neighbors(vid: str, ego_pos: float, tgt_lane_id: str):
    """获取目标车道前后车(id, gap)"""
    lead_id, lead_gap = "", float('inf')
    fol_id,  fol_gap  = "", float('inf')
    for oid in traci.lane.getLastStepVehicleIDs(tgt_lane_id):
        if oid == vid:
            continue
        gap = traci.vehicle.getLanePosition(oid) - ego_pos
        if gap > 0 and gap < lead_gap:
            lead_gap, lead_id = gap, oid
        elif gap <= 0 and abs(gap) < fol_gap:
            fol_gap, fol_id = abs(gap), oid
    return lead_id, (lead_gap if lead_gap < float('inf') else 200.0), \
           fol_id,  (fol_gap  if fol_gap  < float('inf') else 200.0)

def get_accident_broadcast_range(vid: str) -> float:
        """
        事故感知范围（混合交通）：
            - CAV 有序期（广播已激活）：全局广播
            - CAV 突发期 / 人类车：仅局部感知
        """
        if _accident_state["broadcast_active"] and is_vehicle_cav(vid):
            return GLOBAL_V2X_RANGE
        return V2X_RANGE

def get_vehicle_phase(vid: str, veh_pos: float) -> str:
    """
    判断当前车辆所处的事故感知阶段：
      normal   — 事故未发生，或车辆尚未进入感知范围
      sudden   — 事故已发生，但全局广播尚未激活（突发混乱期）
      informed — 全局广播已激活（有序疏散期）
    """
    if not _accident_state["happened"]:
        return "normal"
    dist = ACCIDENT_START - veh_pos
    if dist < -10.0:              # 已通过事故区
        return "normal"
    detect = get_accident_broadcast_range(vid)
    if dist > detect:             # 尚未进入感知范围
        return "normal"
    return "informed" if _accident_state["broadcast_active"] else "sudden"

def estimate_local_density(road_id: str, lane_idx: int, ego_pos: float, window: float = LOCAL_DENSITY_RANGE) -> float:
    """估计局部车流密度（0~1），用于反应延迟与阈值自适应。"""
    if not road_id or road_id.startswith(":") or lane_idx < 0:
        return 0.0
    lane_id = f"{road_id}_{lane_idx}"
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    except traci.exceptions.TraCIException:
        return 0.0

    local_cnt = 0
    for oid in veh_ids:
        if oid in OBSTACLE_IDS:
            continue
        try:
            pos = traci.vehicle.getLanePosition(oid)
        except traci.exceptions.TraCIException:
            continue
        if abs(pos - ego_pos) <= window:
            local_cnt += 1
    return float(np.clip(local_cnt / max(LOCAL_DENSITY_NORM, 1e-6), 0.0, 1.0))

def sample_reaction_delay(phase: str, local_density: float, loss_rate: float) -> float:
    """按阶段采样反应延迟，并用局部密度与丢包率做增益修正。"""
    if phase == "sudden":
        lo, hi = REACTION_DELAY_CAV_SUDDEN
    else:
        lo, hi = REACTION_DELAY_CAV_INFORMED

    mean_delay = max(0.5 * (lo + hi), 1e-3)
    sigma = max(REACTION_DELAY_LOGLN_SIGMA, 1e-3)
    mu = np.log(mean_delay) - 0.5 * sigma * sigma
    delay = float(np.random.lognormal(mu, sigma))
    delay = float(np.clip(delay, lo, hi))

    factor = (
        1.0
        + REACTION_DELAY_DENSITY_GAIN * float(np.clip(local_density, 0.0, 1.0))
        + REACTION_DELAY_LOSS_GAIN * float(np.clip(loss_rate, 0.0, 1.0))
    )
    delay *= factor
    return float(np.clip(delay, REACTION_DELAY_MIN, REACTION_DELAY_MAX))

def check_and_assign_reaction(vid: str, step: int, phase: str, road_id: str, lane_idx: int, veh_pos: float) -> bool:
    """
    首次感知到事故时为该车辆分配随机个体反应延迟；
    若反应延迟已过期返回 True（可开始决策），否则返回 False。
    """
    if vid not in _phase_aware_vids:
        _phase_aware_vids.add(vid)
        _aware_start_steps[vid] = step
        local_density = estimate_local_density(road_id, lane_idx, veh_pos)
        if _perf_metrics["comm_msgs"] > 0:
            loss_rate = _perf_metrics["packet_loss_cnt"] / max(_perf_metrics["comm_msgs"], 1)
        else:
            loss_rate = PACKET_LOSS_RATE
        delay = sample_reaction_delay(phase, local_density, loss_rate)
        _reaction_delays[vid] = step + int(np.ceil(delay / STEP_LEN))
    return step >= _reaction_delays.get(vid, 0)

def try_create_accident(step: int) -> bool:
    """
    从 ACCIDENT_LANE 上距 ACCIDENT_START 最近的车辆中选取最多 2 辆，
    强制移动让它们紧贴在一起（模拟真实的追尾碰撞），然后原地冻结为障碍物。
    找到至少 1 辆返回 True，否则返回 False（下一步继续尝试）。
    """
    try:
        lane_id = f"E0_{ACCIDENT_LANE}"
        candidates = []
        for vid in traci.lane.getLastStepVehicleIDs(lane_id):
            if vid in OBSTACLE_IDS:
                continue
            pos = traci.vehicle.getLanePosition(vid)
            if abs(pos - ACCIDENT_START) <= ACCIDENT_SEARCH_WINDOW:
                candidates.append((pos, vid))
        if not candidates:
            return False
            
        # 按位置降序排列，保证前车在前面
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:2]
        
        base_pos = selected[0][0]
        
        for i, (orig_pos, oid) in enumerate(selected):
            traci.vehicle.setSpeed(oid, 0.0)
            traci.vehicle.setLaneChangeMode(oid, 0)
            traci.vehicle.setSpeedMode(oid, 0)
            
            # 把车移动到紧贴的位置 (车长4.5m，微留0.1m缝隙)
            target_pos = base_pos - i * 4.6
            try:
                traci.vehicle.moveTo(oid, lane_id, target_pos)
            except Exception:
                pass
                
            OBSTACLE_IDS.append(oid)
        return True
    except Exception:
        return False

def get_platoon_members(vid: str, road_id: str, lane_idx: int) -> list:
    """识别与当前车辆处于同一轻量编队/协同簇的连续CAV。"""
    lane_id = f"{road_id}_{lane_idx}"
    ego_pos = traci.vehicle.getLanePosition(vid)
    members = []
    for oid in traci.lane.getLastStepVehicleIDs(lane_id):
        gap = abs(traci.vehicle.getLanePosition(oid) - ego_pos)
        if gap <= PLATOON_HEADWAY:
            members.append(oid)
    return members

def apply_cooperative_yield(ego_vid: str, fol_id: str, fol_gap: float):
    """目标车道后车若为CAV，则主动减速让行，形成协同间隙。混合交通：仅CAV可协同。"""
    if not fol_id:
        return False
    if not is_vehicle_cav(fol_id):
        return False  # 人类车不接受协同请求
    if fol_gap >= COOP_REQUEST_GAP:
        return False
    if fol_id in _coop_state:
        return False
    try:
        cur_speed = traci.vehicle.getSpeed(fol_id)
        ego_speed = traci.vehicle.getSpeed(ego_vid)
        # 协同让行速度控制：后车速度应低于当前速度，并尽量低于本车速度形成间隙
        yield_speed_cap = max(2.0, ego_speed - 1.0)
        target_speed = min(cur_speed - COOP_SLOW_DELTA, yield_speed_cap)
        target_speed = float(np.clip(target_speed, 0.0, cur_speed))
        if target_speed > cur_speed - 0.3:
            target_speed = max(0.0, cur_speed - 0.8)
        traci.vehicle.setSpeed(fol_id, target_speed)
        _coop_state[fol_id] = {
            "ego_vid": ego_vid,
            "request_step": _cur_step,
            "init_speed": cur_speed,
            "responded": False,
            "gap_ready": False,
            "success": False,
        }
        return True
    except Exception:
        return False

def release_cooperative_yield(active_ids: set, step: int, coop_stats: dict):
    """协同让行结束后恢复支援车辆默认控制。"""
    for supporter, info in list(_coop_state.items()):
        ego_vid = info["ego_vid"]
        if supporter not in active_ids or ego_vid not in active_ids:
            if not info["success"]:
                coop_stats["coop_fail_cnt"] += 1
            _coop_state.pop(supporter, None)
            continue
        try:
            supporter_pos = traci.vehicle.getLanePosition(supporter)
            ego_pos = traci.vehicle.getLanePosition(ego_vid)
            supporter_speed = traci.vehicle.getSpeed(supporter)
            if not info["responded"] and supporter_speed <= max(0.0, info["init_speed"] - 0.5):
                info["responded"] = True
                coop_stats["coop_response_times"].append((step - info["request_step"]) * STEP_LEN)
            if not info["gap_ready"] and abs(supporter_pos - ego_pos) >= COOP_MIN_GAP:
                info["gap_ready"] = True
                coop_stats["coop_gap_build_times"].append((step - info["request_step"]) * STEP_LEN)
            # 解除条件：两者纵向拉开安全距离、请求者已经完成或取消了变道意图、超时（15秒卡死保护）
            if abs(supporter_pos - ego_pos) > COOP_RECOVER_DIST or (ego_vid not in _lc_state) or ((step - info["request_step"]) * STEP_LEN > 15.0):
                traci.vehicle.setSpeed(supporter, -1)
                if not info["success"]:
                    coop_stats["coop_fail_cnt"] += 1
                _coop_state.pop(supporter, None)
        except Exception:
            if not info["success"]:
                coop_stats["coop_fail_cnt"] += 1
            _coop_state.pop(supporter, None)

def mark_cooperative_success(ego_vid: str, step: int, coop_stats: dict):
    """当ego开始执行换道时，将其对应协同让行标记为成功。"""
    for supporter, info in list(_coop_state.items()):
        if info["ego_vid"] != ego_vid or info["success"]:
            continue
        info["success"] = True
        coop_stats["coop_success_cnt"] += 1
        if not info["gap_ready"]:
            coop_stats["coop_gap_build_times"].append((step - info["request_step"]) * STEP_LEN)

# ====================== 感知模型函数 ======================
def _add_noise(val: float, std: float) -> float:
    """乘法高斯噪声，结果截断至非负值"""
    return max(0.0, val * (1.0 + np.random.normal(0.0, std)))

def perc_speed(vid: str) -> float:
    """读取邻车速度并施加 ±5% 高斯噪声（SPEED_NOISE_STD·1σ）"""
    try:
        spd = traci.vehicle.getSpeed(vid)
    except traci.exceptions.TraCIException:
        spd = 0.0
    return _add_noise(spd, SPEED_NOISE_STD)

def sample_perception(vid: str, ego_pos: float, tgt_lane_id: str) -> tuple:
    """
    带延迟 + 噪声 + 丢包的邻居感知函数（CAV 专用）:
       1. 若发生丢包，保留最后一次已知观测
       2. 施加高斯噪声写入感知缓冲区
       3. 返回延迟的缓存观测
    """
    _perf_metrics["comm_msgs"] += 1
    packet_lost = (np.random.rand() < PACKET_LOSS_RATE)
    if packet_lost:
        _perf_metrics["packet_loss_cnt"] += 1

    lead_id, lead_gap, fol_id, fol_gap = get_neighbors(vid, ego_pos, tgt_lane_id)
    cur_obs = (
        _cur_step,
        lead_id,
        _add_noise(lead_gap, DIST_NOISE_STD),
        fol_id,
        _add_noise(fol_gap, DIST_NOISE_STD),
    )
    buf = _perc_buf.setdefault(vid, [])
    
    # 未丢包或首次强行获取，才能更新最新观测
    if not packet_lost or len(buf) == 0:
        buf.append(cur_obs)
        
    # 保留最多 delay+3 帧，避免缓冲区无限增长
    keep_n = PERC_DELAY_STEPS + 3
    if len(buf) > keep_n:
        del buf[:-keep_n]
    delayed_idx = max(0, len(buf) - 1 - PERC_DELAY_STEPS)
    obs = buf[delayed_idx]
    return obs[1], obs[2], obs[3], obs[4]

def get_lc_profile(vid: str) -> tuple:
    """返回三阶段时长(准备, 执行, 稳定)，并确保执行阶段满足横向加速度约束"""
    prep_dur, exec_nominal, stab_dur = LC_PREP_CAV, LC_DURATION_CAV, LC_STAB_CAV
    exec_dur = max(exec_nominal, MIN_LC_EXEC_DURATION)
    return prep_dur, exec_dur, stab_dur

def estimate_lat_acc(exec_dur: float) -> float:
    """按正弦横向速度剖面估计峰值横向加速度"""
    return float((np.pi ** 2) * LANE_WIDTH / (2.0 * max(exec_dur, 1e-6) ** 2))

def compute_ttc(gap: float, rel_speed: float) -> float:
    """TTC估计：closing speed 非正时视为无碰撞风险。"""
    if rel_speed <= 1e-3:
        return float("inf")
    return max(gap, 0.0) / rel_speed

def dynamic_min_gap(ego_speed: float) -> float:
    """随速度变化的动态最小安全间距。"""
    return max(MIN_SAFE_GAP, 2.0 + 0.35 * max(ego_speed, 0.0))

def release_emergency_braking_control(vid: str):
    """退出紧急制动覆盖，恢复SUMO默认纵向控制。"""
    if vid not in _emergency_brake_vids:
        return
    try:
        traci.vehicle.setSpeed(vid, -1)
    except traci.exceptions.TraCIException:
        pass
    _emergency_brake_vids.discard(vid)

def get_obstacle_anchor_position(active_ids: set) -> float:
    """返回事故车道上后方障碍锚点位置（来车最先接触到的障碍）。"""
    positions = []
    for oid in OBSTACLE_IDS:
        if oid not in active_ids:
            continue
        try:
            if traci.vehicle.getRoadID(oid) != "E0":
                continue
            if traci.vehicle.getLaneIndex(oid) != ACCIDENT_LANE:
                continue
            positions.append(traci.vehicle.getLanePosition(oid))
        except traci.exceptions.TraCIException:
            continue
    if positions:
        return float(min(positions))
    return ACCIDENT_START

def compute_stop_distance(speed: float) -> float:
    """基于停车距离模型估计最小停车需求距离。"""
    v = max(float(speed), 0.0)
    dec = max(EMERGENCY_EFF_DEC, 1e-3)
    return (
        v * EMERGENCY_REACT_TIME
        + (v * v) / (2.0 * dec)
        + EMERGENCY_MARGIN
    )

def compute_safe_speed_by_distance(distance: float) -> float:
    """由剩余距离反推安全速度包络。"""
    dec = max(EMERGENCY_EFF_DEC, 1e-3)
    remain = max(float(distance) - EMERGENCY_MARGIN, 0.0)
    return float(np.sqrt(max(2.0 * dec * remain, 0.0)))

def apply_emergency_braking_coverage(vid: str, road_id: str, lane_idx: int, veh_pos: float, obstacle_anchor: float) -> bool:
    """
    事故车道障碍前紧急制动覆盖：
       1) d <= d_stop + buffer 时触发
       2) 速度受 v_safe(d) 包络限制
       3) 覆盖触发后优先级高于换道博弈
    """
    if road_id != "E0" or lane_idx != ACCIDENT_LANE:
        release_emergency_braking_control(vid)
        return False

    dist = float(obstacle_anchor - veh_pos)
    if dist <= 0.0 or dist > EMERGENCY_MAX_ZONE:
        release_emergency_braking_control(vid)
        return False

    try:
        cur_speed = traci.vehicle.getSpeed(vid)
    except traci.exceptions.TraCIException:
        release_emergency_braking_control(vid)
        return False

    stop_need = compute_stop_distance(cur_speed)
    force_cover = (dist <= EMERGENCY_FORCE_BRAKE_DIST)
    if (not force_cover) and dist > stop_need + EMERGENCY_TRIGGER_BUFFER:
        release_emergency_braking_control(vid)
        return False

    target_speed = min(cur_speed, compute_safe_speed_by_distance(dist))
    if force_cover:
        # 近障碍进入强制刹停区，按距离线性压速，避免末端激进并线。
        force_cap = max(0.0, 0.12 * max(dist - EMERGENCY_MARGIN, 0.0))
        target_speed = min(target_speed, force_cap)
    if dist <= EMERGENCY_MARGIN + 1.0:
        target_speed = 0.0

    try:
        traci.vehicle.setSpeed(vid, float(np.clip(target_speed, 0.0, max(cur_speed, 0.0))))
    except traci.exceptions.TraCIException:
        return False

    if dist <= max(EMERGENCY_PREP_ABORT_DIST, EMERGENCY_NO_LC_DIST):
        state = _lc_state.get(vid)
        if state and state.get("phase") == "prepare":
            _lc_state.pop(vid, None)
    _emergency_brake_vids.add(vid)
    return True

def lanechange_hard_safety_check(vid: str, road_id: str, tgt_lane: int, phase: str) -> bool:
    """基于真实邻车信息的硬安全检查，用于执行前兜底。"""
    if not vid or not road_id or road_id.startswith(":"):
        return False
    tgt_lid = f"{road_id}_{tgt_lane}"
    try:
        ego_pos = traci.vehicle.getLanePosition(vid)
        ego_spd = traci.vehicle.getSpeed(vid)
    except traci.exceptions.TraCIException:
        return False

    lead_id, lead_gap, fol_id, fol_gap = get_neighbors(vid, ego_pos, tgt_lid)
    dyn_gap = dynamic_min_gap(ego_spd)
    if lead_gap < dyn_gap or fol_gap < dyn_gap:
        return False

    lead_spd = ego_spd
    fol_spd = 0.0
    if lead_id:
        try:
            lead_spd = traci.vehicle.getSpeed(lead_id)
        except traci.exceptions.TraCIException:
            lead_spd = ego_spd
    if fol_id:
        try:
            fol_spd = traci.vehicle.getSpeed(fol_id)
        except traci.exceptions.TraCIException:
            fol_spd = 0.0

    ttc_front = compute_ttc(lead_gap, max(ego_spd - lead_spd, 0.0))
    ttc_rear = compute_ttc(fol_gap, max(fol_spd - ego_spd, 0.0))
    ttc_min = min(ttc_front, ttc_rear)
    ttc_hard_min = TTC_HARD_MIN_SUDDEN if phase == "sudden" else TTC_HARD_MIN_INFORMED
    return ttc_min >= (ttc_hard_min + 0.30)

def safety_from_gap_ttc(gap: float, rel_speed: float) -> float:
    """基于间距与TTC的安全评分（0~1，越大越安全）。"""
    gap_term = float(np.clip((gap - MIN_SAFE_GAP) / 25.0, 0.0, 1.0))
    if rel_speed <= 1e-3:
        ttc_term = 1.0
    else:
        ttc_term = float(np.clip((gap / rel_speed) / 4.0, 0.0, 1.0))
    return 0.55 * gap_term + 0.45 * ttc_term

# ====================== 博弈核心（解耦原子特征 + 可学习权重） ======================

# 可学习权重（IRL 目标）：[speed, urgency, pressure, safe, coop, social, density, lc_cost]
# 突发期 vs 有序期共享结构，但数值可以不同
PAYOFF_WEIGHTS = {
    "sudden":  np.array([0.25, 0.15, 0.08, 0.30, 0.10, 0.05, 0.05, 1.0]),
    "informed": np.array([0.30, 0.10, 0.06, 0.25, 0.12, 0.05, 0.05, 1.0]),
}

# 特征名称（对应 IRL 的 feature template）
FEATURE_NAMES = ["speed", "urgency", "pressure", "safe", "coop", "social", "density", "lc_cost"]

# 后车收益函数权重: [safety, speed, comfort, coop, gap_safe]
# 每种类型对应不同博弈结构（改进 #1：博弈结构自适应）
#   Type 0 自私型 → 静态 Nash（双方同时决策，最大化个体收益）
#   Type 1 合作型 → Stackelberg（本车 Leader，后车最优反应）
#   Type 2 高度合作型 → Nash 合作（最大化联合收益）
FOLLOWER_TYPE_WEIGHTS = [
    np.array([0.45, 0.35, 0.20, 0.00, 0.00]),  # Type 0: 自私型 (20%)
    np.array([0.40, 0.25, 0.15, 0.20, 0.00]),  # Type 1: 合作型 (60%)
    np.array([0.35, 0.15, 0.15, 0.35, 0.00]),  # Type 2: 高度合作型 (20%)
]
FOLLOWER_TYPE_SHARES = [0.20, 0.60, 0.20]

# 混合策略温度系数（改进 #3）：越低越接近纯策略，越高越随机
SOFTMAX_TEMPERATURE = 0.15

# 在线学习参数（改进 #5）
ONLINE_LEARNING_RATE = 0.003  # TD 微调步长


def _softmax(x: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    """温度 softmax：temperature 越低越接近 argmax。"""
    x_adj = (x - x.max()) / max(temperature, 1e-6)
    e = np.exp(np.clip(x_adj, -20.0, 20.0))
    return e / e.sum()

# IRL 特征日志（run_once 每次仿真开始时清空，结束时可供外部读取）
_irl_feature_log: list = []  # [(payoff, feat_dict), ...]
_irl_feature_log_snapshot: list = []  # run_once 结束时的快照

# 在线学习记录（改进 #5）
_online_learning_memory: list = []  # [(weights_before, features_chosen, success, hard_brake), ...]


def get_adaptive_weights(phase: str, local_density: float, urgency: float) -> np.ndarray:
    """
    改进 #4：权重自适应 —— 随场景动态调整。

    堵车/紧迫时 → 安全权重、社会冲击权重 ↑，速度权重 ↓
    """
    base = PAYOFF_WEIGHTS[phase].copy()

    # 安全权重随密度和紧迫度上升
    base[3] += 0.10 * local_density + 0.05 * urgency  # safe ↑

    # 社会冲击在高密度时更敏感
    base[5] += 0.05 * local_density  # social ↑

    # 紧迫度高时降低速度追求，优先安全
    base[0] = max(0.05, base[0] - 0.10 * urgency)  # speed ↓

    return np.clip(base, 0.01, 2.0)


def online_update_weights(success: bool, hard_brake: bool):
    """
    改进 #5：在线 TD 微调 —— 每次换道执行后基于结果即时更新权重。

    success=True, hard_brake=False → 强化当前权重
    hard_brake=True → 惩罚：提升安全权重，降低社会冲击容忍度
    """
    if not _online_learning_memory:
        return
    _online_learning_memory.pop()  # 消费本次记忆

    lr = ONLINE_LEARNING_RATE
    if success and not hard_brake:
        # 成功换道：轻量强化安全+协同维度
        PAYOFF_WEIGHTS["informed"][3] += lr * 0.1  # safe
        PAYOFF_WEIGHTS["informed"][4] += lr * 0.1  # coop
    elif hard_brake:
        # 换道导致急刹：惩罚
        PAYOFF_WEIGHTS["informed"][3] += lr * 0.8  # safe ↑↑
        PAYOFF_WEIGHTS["informed"][5] -= lr * 0.8  # social ↓↓
    else:
        # 失败：适度调整
        PAYOFF_WEIGHTS["informed"][3] += lr * 0.2  # safe ↑

    PAYOFF_WEIGHTS["informed"] = np.clip(PAYOFF_WEIGHTS["informed"], 0.01, 2.0)


def compute_features(vid, ego_speed, vmax, lead_spd, fol_spd, lead_gap, fol_gap,
                     cur_lead_gap, urgency, cur_lane_pressure, coop_bonus,
                     local_density, delta_vs=(2.0, 0.0, -3.0), phase="informed") -> dict:
    """
    提取换道决策的 8 维原子特征向量。

    特征: [speed, urgency, pressure, safe, coop, social, density, lc_cost]

    参数 delta_vs 支持连续（动态）后车反应（改进 #3）。
    social 字段量化换道对后车 TTC 的冲击（改进 #4）。
    density 从阈值移入特征空间（改进 #5）。
    """
    features = {}

    # 计算社会影响基线：后车当前距目标车道前车的 TTC
    if lead_gap < 200.0 and fol_gap < 200.0 and lead_spd > 1.0 and fol_spd > 1.0:
        fol_lead_gap = fol_gap + lead_gap  # 后车到目标车道前车的距离
        fol_rel_leader = max(fol_spd - lead_spd, 0.1)
        fol_old_ttc = fol_lead_gap / fol_rel_leader if fol_rel_leader > 1e-3 else 10.0
    else:
        fol_old_ttc = 10.0

    for fa, delta_v in enumerate(delta_vs):
        fol_spd_new = float(np.clip(fol_spd + delta_v, 0.0, vmax))
        front_rel = max(ego_speed - lead_spd, 0.0)
        rear_rel = max(fol_spd_new - ego_speed, 0.0)

        # ── 换道 (action=0) ──
        safe_lc = min(
            safety_from_gap_ttc(lead_gap, front_rel),
            safety_from_gap_ttc(fol_gap, rear_rel),
        )

        # 社会影响：换道后后车 TTC 缩减比例
        fol_rel_ego = max(fol_spd_new - ego_speed, 0.1)
        fol_new_ttc = fol_gap / fol_rel_ego if fol_rel_ego > 1e-3 else 10.0
        social = min(max(fol_old_ttc - fol_new_ttc, 0.0) / max(fol_old_ttc, 0.1), 1.0)

        features[(0, fa)] = np.array([
            float(np.clip(lead_spd / max(vmax, 1e-3), 0.0, 1.0)),  # speed: 目标车道前车速度比
            urgency,                                                   # urgency: 距事故区紧迫度
            cur_lane_pressure,                                         # pressure: 当前车道压力
            safe_lc,                                                   # safe: 换道后综合 TTC 安全
            coop_bonus + (0.12 if fa == 2 else 0.0),                   # coop: 后车协同奖励
            social,                                                    # social: 对后车冲击
            local_density,                                             # density: 局部密度
            1.0,                                                       # lc_cost: 换道代价
        ])

        # ── 不换道 (action=1) ──
        safe_keep = safety_from_gap_ttc(
            cur_lead_gap, max(ego_speed - min(lead_spd, ego_speed), 0.0)
        )
        features[(1, fa)] = np.array([
            float(np.clip(ego_speed / max(vmax, 1e-3), 0.0, 1.0)),  # speed: 自车速度比
            0.0,                                                       # urgency: 不换道无紧迫度收益
            0.0,                                                       # pressure: 不换道无压力收益
            safe_keep,                                                 # safe: 当前车道安全
            0.05,                                                      # coop: 基础协同值
            0.0,                                                       # social: 不换道无社会冲击
            local_density,                                             # density: 局部密度
            0.0,                                                       # lc_cost: 无换道代价
        ])

    return features


def compute_payoff(vid, ego_speed, ego_pos, lead_id, lead_gap, fol_id, fol_gap,
                   cur_lead_gap, road_id, cur_lane, phase="informed") -> np.ndarray:
    """
    2×3 收益矩阵: ego{换道/不换道} × follower{加速/保持/减速}

    改进 #3: delta_v 根据后车间距/速度动态计算（连续后车反应）
    改进 #4: 在 compute_features 内部计算 social impact
    改进 #5: density 作为特征传入，不再作为阈值修正
    """
    vmax = cached_max_speed(vid)
    lead_spd = perc_speed(lead_id) if lead_id else vmax
    fol_spd = perc_speed(fol_id) if fol_id else 0.0

    detect = V2X_RANGE if phase == "sudden" else get_accident_broadcast_range(vid)
    dist = max(ACCIDENT_START - ego_pos, 0.1)
    safe_braking_dist = vmax * 1.0 + (vmax ** 2) / 5.0 + 50.0
    urgency_range = min(detect, max(safe_braking_dist, 100.0))
    urgency = float(np.clip(1.0 - dist / urgency_range, 0.0, 1.0))
    cur_lane_pressure = float(np.clip((30.0 - cur_lead_gap) / 30.0, 0.0, 1.0))
    coop_bonus = 0.18 if fol_id else 0.0

    # 局部密度
    local_density = estimate_local_density(road_id, cur_lane, ego_pos)

    # 改进 #4：自适应权重（密度/紧迫度 → 安全↑ 速度↓）
    weights = get_adaptive_weights(phase, local_density, urgency)

    # 连续后车反应：根据间距/速度动态计算 delta_v（改进 #3）
    if fol_id:
        fol_max_spd = traci.vehicle.getMaxSpeed(fol_id) if hasattr(traci.vehicle, 'getMaxSpeed') else vmax
        fol_speed_ratio = fol_spd / max(fol_max_spd, 1e-3)
        # 制动强度：间距越紧制动越强
        if fol_gap < COOP_MIN_GAP:
            brake_dv = -min(4.5, max(1.5, 6.0 / max(fol_gap, 1.0)))
        else:
            brake_dv = -1.5
        # 加速强度：空间越大、速度越低，加速越强
        if fol_gap > COOP_REQUEST_GAP and fol_speed_ratio < 0.8:
            accel_dv = min(2.6, 0.5 + 2.0 * min(fol_gap / COOP_REQUEST_GAP, 1.0))
        else:
            accel_dv = 0.5
    else:
        accel_dv, brake_dv = 2.0, -3.0

    delta_vs = (accel_dv, 0.0, brake_dv)

    feats = compute_features(vid, ego_speed, vmax, lead_spd, fol_spd,
                             lead_gap, fol_gap, cur_lead_gap,
                             urgency, cur_lane_pressure, coop_bonus,
                             local_density, delta_vs, phase)

    payoff = np.zeros((2, 3))
    for action in (0, 1):
        for fa in (0, 1, 2):
            v = np.dot(weights, feats[(action, fa)])
            if action == 0:
                v -= LANE_CHANGE_COST
            payoff[action, fa] = v

    # IRL 日志：记录决策点的特征（由外部清空）
    _irl_feature_log.append((payoff, feats))
    return payoff


def compute_follower_payoff(fol_spd: float, fol_gap: float, ego_speed: float,
                             lead_spd: float, fol_max: float, weight_type: int = 1,
                             phase: str = "informed") -> np.ndarray:
    """
    后车（follower）的收益函数。

    返回 (2, 3) 矩阵: ego_action{换道,不换道} × follower_action{加速,保持,减速}

    特征: [safety, speed, comfort, coop, gap_safe]
    - safety: 后车追尾前车 TTC 评分
    - speed: 后车速度有效值
    - comfort: 制动不舒适代价（负贡献）
    - coop: 协同让行社会奖励（仅 CAV 环境）
    - gap_safe: 间距是否低于安全底线
    """
    weights = FOLLOWER_TYPE_WEIGHTS[weight_type]

    payoff = np.zeros((2, 3))
    for ea in (0, 1):  # ego: 0=换道(切入), 1=保持(不切入)
        for fa, delta_v in enumerate([2.0, 0.0, -3.0]):
            fol_spd_new = float(np.clip(fol_spd + delta_v, 0.0, fol_max))

            if ea == 0:
                # 本车换道切入：后车前方变为本车
                gap_to_leader = fol_gap
                rel_to_leader = max(fol_spd_new - ego_speed, 0.0)
                safe_score = safety_from_gap_ttc(gap_to_leader, rel_to_leader)
            else:
                # 本车不换道：后车前方不变，安全无新威胁
                safe_score = 1.0

            speed_ratio = float(np.clip(fol_spd_new / max(fol_max, 1e-3), 0.0, 1.0))
            comfort = max(0.0, -delta_v / 4.5)  # 制动强度归一化 0~1
            coop = 1.0 if (ea == 0 and fa == 2) else 0.0  # 本车切入+后车减速=协同
            gap_safe = 1.0  # 可由更复杂的间距约束计算

            payoff[ea, fa] = float(np.dot(
                weights, [safe_score, speed_ratio, comfort, coop, gap_safe]))

    return payoff


def _solve_static_nash(ego_payoff: np.ndarray, fol_payoff: np.ndarray) -> np.ndarray:
    """
    静态 Nash 博弈（Type 0 自私型专用）：双方同时决策，无领导者。

    使用混合策略均衡——后车以 softmax 概率选择动作，
    本车据此计算期望。迭代至稳定。
    """
    ego_strat = np.array([0.5, 0.5])
    for _ in range(4):  # 4 轮迭代足够收敛（原 8 轮）
        fol_expected = fol_payoff.T @ ego_strat
        fol_strat = _softmax(fol_expected)
        # 本车对后车混合策略的期望
        ego_expected = ego_payoff @ fol_strat
        ego_strat = _softmax(ego_expected)
    return ego_expected  # 本车两个动作的期望收益


def _solve_stackelberg_per_type(ego_payoff: np.ndarray,
                                  fol_payoff: np.ndarray) -> np.ndarray:
    """
    Stackelberg 博弈（Type 1 合作型）：本车领导者，后车最优反应（混合策略）。
    改进 #3：softmax 替代 argmax，模拟行为随机性。
    """
    expected = np.zeros(2)
    for ea in (0, 1):
        fol_action_probs = _softmax(fol_payoff[ea])
        expected[ea] = float(np.dot(ego_payoff[ea], fol_action_probs))
    return expected


def _solve_nash_cooperative(ego_payoff: np.ndarray,
                              fol_payoff: np.ndarray) -> np.ndarray:
    """
    Nash 合作博弈（Type 2 高度合作型）：最大化联合收益。
    改进 #2：CAV↔CAV 合作，双方一起找最优组合。
    """
    joint = ego_payoff + fol_payoff  # 2×3 联合收益
    expected = np.zeros(2)
    for ea in (0, 1):
        # 后车选使联合收益最大的动作（softmax 软化）
        joint_probs = _softmax(joint[ea])
        expected[ea] = float(np.dot(ego_payoff[ea], joint_probs))
    return expected


def solve_hybrid_game(ego_payoff: np.ndarray, fol_id: str, ego_speed: float,
                       fol_gap: float, lead_spd: float, phase: str = "informed") -> np.ndarray:
    """
    混合博弈求解器（改进 #1）：根据后车类型选择博弈结构。

    Type 0 (20%): 静态 Nash  — 各算各的，同时决策
    Type 1 (60%): Stackelberg — 本车 Leader，后车最优反应
    Type 2 (20%): Nash 合作  — 最大化联合收益（CAV↔CAV）
    """
    expected = np.zeros(2)

    try:
        fol_spd = traci.vehicle.getSpeed(fol_id)
        fol_max = cached_max_speed(fol_id)
    except traci.exceptions.TraCIException:
        fol_spd = 0.0
        fol_max = 33.33

    for t_idx, share in enumerate(FOLLOWER_TYPE_SHARES):
        fol_payoff = compute_follower_payoff(
            fol_spd, fol_gap, ego_speed, lead_spd, fol_max, t_idx, phase)

        if t_idx == 0:
            ea_expected = _solve_static_nash(ego_payoff, fol_payoff)
        elif t_idx == 1:
            ea_expected = _solve_stackelberg_per_type(ego_payoff, fol_payoff)
        else:  # t_idx == 2
            ea_expected = _solve_nash_cooperative(ego_payoff, fol_payoff)

        expected += share * ea_expected

    return expected

def get_follower_prior(fol_id: str, fol_gap: float,
                        level_k: int = 0) -> np.ndarray:
    """
    后车行为先验预测（改进 #1, #3）。

    改进 #1（对抗性偏差）：后车=人类 → 加速堵位概率↑，减速让行概率↓
    改进 #3（互惠记忆）：曾让行的人类车 → 合作先验↑

    返回 [P(加速), P(保持), P(减速)]
    """
    if not fol_id:
        return np.array([0.1, 0.5, 0.4])

    try:
        fol_spd = traci.vehicle.getSpeed(fol_id)
        fol_max = cached_max_speed(fol_id)
    except traci.exceptions.TraCIException:
        fol_spd = 0.0
        fol_max = 33.33

    speed_ratio = fol_spd / max(fol_max, 1e-3)
    if fol_gap < COOP_MIN_GAP or speed_ratio > 0.85:
        base = np.array([0.1, 0.3, 0.6], dtype=float)
    elif fol_gap > COOP_REQUEST_GAP and speed_ratio < 0.5:
        base = np.array([0.4, 0.4, 0.2], dtype=float)
    else:
        base = np.array([0.2, 0.5, 0.3], dtype=float)

    # 在线行为修正
    hist = _fol_acc_hist.get(fol_id, [])
    if hist:
        recent_acc = float(np.mean(hist[-FOLLOWER_BEHAV_WINDOW:]))
        if recent_acc <= -0.2:
            base += np.array([-0.06, -0.02, 0.08])
        elif recent_acc >= 0.2:
            base += np.array([0.08, -0.03, -0.05])

    # Level-k 认知层级调整
    if level_k == 1:
        base += np.array([-0.05, -0.05, 0.10])
    elif level_k == 2:
        base += np.array([0.05, 0.10, -0.15])

    # 改进 #1：对抗性偏差 — 后车=人类 → 不配合倾向
    if fol_id and not is_vehicle_cav(fol_id):
        # 人类车缺乏合作动机，加速堵位或保持原速，很少主动减速让行
        base += np.array([0.10, 0.00, -0.10])  # acc ↑, brake ↓

    # 改进 #3：互惠记忆 — 历史上合作过的车提高合作先验
    coop_score = _human_coop_memory.get(fol_id, None)
    if coop_score is not None:
        # cooperation_score 越高 → 越倾向减速让行
        extra_coop = (coop_score - 0.5) * 0.20  # [-0.06, +0.06] range
        base += np.array([-extra_coop, 0.0, extra_coop])

    base = np.clip(base, 0.01, None)
    base /= np.sum(base)
    return base

def decide_lanechange(vid, cur_lane, tgt_lane, road_id, phase="informed") -> float:
    """
    换道决策函数：基于同时博弈（2×3 收益矩阵）计算期望收益。
    """
    # 自车速度施加感知噪声 ±5%（模拟传感器测量误差）
    try:
        ego_spd = _add_noise(traci.vehicle.getSpeed(vid), SPEED_NOISE_STD)
    except traci.exceptions.TraCIException:
        ego_spd = 0.0
    ego_pos = traci.vehicle.getLanePosition(vid)
    detect = V2X_RANGE if phase == "sudden" else get_accident_broadcast_range(vid)
    dist   = ACCIDENT_START - ego_pos
    if not (-10.0 < dist < detect):
        return -float('inf')
    tgt_lid = f"{road_id}_{tgt_lane}"
    cur_lid = f"{road_id}_{cur_lane}"

    # CAV：带 0.1s 延迟 + 间距噪声 ±3% 的感知模型
    lead_id, lead_gap, fol_id, fol_gap = sample_perception(vid, ego_pos, tgt_lid)
    _, cur_lead_gap, _, _ = sample_perception(vid, ego_pos, cur_lid)

    dyn_gap = dynamic_min_gap(ego_spd)
    if lead_gap < dyn_gap or fol_gap < dyn_gap:
        return -float('inf')

    # 硬安全门控：若前/后向TTC过小，直接拒绝换道
    lead_spd_est = perc_speed(lead_id) if lead_id else ego_spd
    fol_spd_est = perc_speed(fol_id) if fol_id else 0.0
    front_rel = max(ego_spd - lead_spd_est, 0.0)
    rear_rel = max(fol_spd_est - ego_spd, 0.0)
    ttc_front = compute_ttc(lead_gap, front_rel)
    ttc_rear = compute_ttc(fol_gap, rear_rel)
    ttc_min = min(ttc_front, ttc_rear)
    ttc_hard_min = TTC_HARD_MIN_SUDDEN if phase == "sudden" else TTC_HARD_MIN_INFORMED
    if ttc_min < ttc_hard_min:
        return -float('inf')

    # 计算本车（ego）收益矩阵（改进 #4：自适应权重内置于 compute_payoff）
    payoff = compute_payoff(vid, ego_spd, ego_pos, lead_id, lead_gap,
                            fol_id, fol_gap, cur_lead_gap,
                            road_id, cur_lane, phase)

    if fol_id and is_vehicle_cav(fol_id):
        # 改进 #1, #2, #3：混合博弈 — 按后车类型选博弈结构 + 混合策略
        lead_spd_est = perc_speed(lead_id) if lead_id else ego_spd
        expected = solve_hybrid_game(payoff, fol_id, ego_spd, fol_gap,
                                     lead_spd_est, phase)
    else:
        # 无后车 或 后车是人类：概率预测（不博弈）
        prior = get_follower_prior(fol_id, fol_gap, level_k=0)
        expected = payoff @ prior

    gain = expected[0] - expected[1]
    base_min_gain = GAME_MIN_GAIN_SUDDEN if phase == "sudden" else GAME_MIN_GAIN_INFORMED

    # 改进 #2：自适应保守度 — 低渗透率降低阈值，使 CAV 更果断
    # 理由：周围都是人类车时，犹豫只会错过间隙；高渗透率时可以更挑剔
    adaptive_factor = 1.0 - 0.6 * (1.0 - CAV_PENETRATION)  # 0%→0.4, 100%→1.0
    min_gain = base_min_gain * adaptive_factor

    if gain > min_gain:
        _online_learning_memory.append(PAYOFF_WEIGHTS[phase].copy())
        return float(expected[0])
    return -float('inf')

def decide_platoon_lanechange(platoon: list, cur_lane: int, tgt_lane: int, road_id: str, phase: str = "informed") -> float:
    """轻量编队协调：只要编队头车满足条件，其余CAV跟随同一策略。"""
    if not platoon:
        return -float('inf')

    tgt_lid = f"{road_id}_{tgt_lane}"
    # 严格检查编队中每一辆车的安全性
    for obj_id in platoon:
        veh_pos = traci.vehicle.getLanePosition(obj_id)
        _, lead_gap, _, fol_gap = get_neighbors(obj_id, veh_pos, tgt_lid)
        if lead_gap < MIN_SAFE_GAP or fol_gap < MIN_SAFE_GAP:
            return -float('inf')

    # 头车为 position 最大的车辆
    leader = max(platoon, key=lambda v: traci.vehicle.getLanePosition(v))
    return decide_lanechange(leader, cur_lane, tgt_lane, road_id, phase)


# ====================== 事故障碍物 ======================
def manage_obstacles():
    """每步强制保持事故车辆静止，并清除已离场的障碍物记录。"""
    active = set(traci.vehicle.getIDList())
    for oid in list(OBSTACLE_IDS):
        if oid in active:
            traci.vehicle.setSpeed(oid, 0.0)
            traci.vehicle.setLaneChangeMode(oid, 0)
            traci.vehicle.setSpeedMode(oid, 0)
        else:
            OBSTACLE_IDS.remove(oid)

def apply_speed_limit_vehicle(vid: str, veh_pos: float):
    """对事故预警区[SLOW_ZONE_START, ACCIDENT_END]内的车辆单独限速，离开后恢复原车型最高速"""
    try:
        if SLOW_ZONE_START <= veh_pos <= ACCIDENT_END:
            if traci.vehicle.getMaxSpeed(vid) > SLOW_SPEED:
                traci.vehicle.setMaxSpeed(vid, SLOW_SPEED)
        elif veh_pos > ACCIDENT_END:
            vtype_max = traci.vehicletype.getMaxSpeed(traci.vehicle.getTypeID(vid))
            if traci.vehicle.getMaxSpeed(vid) < vtype_max:
                traci.vehicle.setMaxSpeed(vid, vtype_max)
    except traci.exceptions.TraCIException:
        pass

def apply_normal_cruise_control(vid: str):
    """全CAV常态巡航：基于时距与相对速度的轻量CACC控制。"""
    try:
        cur_speed = traci.vehicle.getSpeed(vid)
        vmax = traci.vehicle.getMaxSpeed(vid)
        leader = traci.vehicle.getLeader(vid, 200.0)

        if leader:
            lead_id, gap = leader
            try:
                lead_speed = traci.vehicle.getSpeed(lead_id)
            except traci.exceptions.TraCIException:
                lead_speed = cur_speed

            desired_gap = MIN_SAFE_GAP + NORMAL_HEADWAY * max(cur_speed, 0.0)
            gap_error = gap - desired_gap
            rel_speed = lead_speed - cur_speed
            target_speed = (
                cur_speed
                + NORMAL_GAP_GAIN * gap_error
                + NORMAL_REL_SPEED_GAIN * rel_speed
            )

            if gap < desired_gap * 0.7:
                target_speed = min(target_speed, lead_speed - 0.8)
        else:
            target_speed = cur_speed + NORMAL_FREEFLOW_GAIN * (vmax - cur_speed)

        target_speed = float(np.clip(target_speed, 0.0, vmax))
        traci.vehicle.setSpeed(vid, target_speed)
        _normal_ctrl_vids.add(vid)
    except traci.exceptions.TraCIException:
        pass

def release_normal_cruise_control(vid: str):
    """退出常态巡航控制，恢复SUMO默认纵向控制。"""
    if vid not in _normal_ctrl_vids:
        return
    try:
        traci.vehicle.setSpeed(vid, -1)
    except traci.exceptions.TraCIException:
        pass
    _normal_ctrl_vids.discard(vid)

# ====================== 单次仿真 ======================
def run_once(n_cav: int, label: str, use_gui: bool = False) -> dict:
    global _cur_step
    # 每次仿真开始前清空所有全局状态
    _lc_state.clear()
    _last_lc_step.clear()
    _perc_buf.clear()
    _coop_state.clear()
    _acc_buf.clear()
    _fol_acc_hist.clear()
    _reaction_delays.clear()
    _aware_start_steps.clear()
    _phase_aware_vids.clear()
    _normal_ctrl_vids.clear()
    _emergency_brake_vids.clear()
    OBSTACLE_IDS.clear()
    _irl_feature_log.clear()
    _online_learning_memory.clear()
    _human_coop_memory.clear()
    _max_speed_cache.clear()
    for k in _perf_metrics:
        _perf_metrics[k] = 0 if isinstance(_perf_metrics[k], int) else 0.0
    _accident_state["happened"]         = False
    _accident_state["time_actual"]      = -1.0
    _accident_state["broadcast_active"] = False
    _cur_step = 0
    rou_file = gen_rou_xml(n_cav)
    # 用 label 生成独立输出文件，避免多实验 tripinfo 冲突
    safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    tripinfo_file = f"tripinfo_{safe_label}.xml"
    lc_file = f"lanechanges_{safe_label}.xml"
    fcd_file = f"fcd_{safe_label}.xml"
    exe      = "sumo-gui" if use_gui else "sumo"
    cmd      = [exe, "-c", "accident_highway.sumocfg",
                "--route-files", rou_file,
                "--lateral-resolution", "0.8",
                "--collision.action", "warn",
                "--no-warnings", "true",
                "--no-step-log", "true",
                "--tripinfo-output", tripinfo_file,
                "--lanechange-output", lc_file,
                "--fcd-output", fcd_file,
                "--gui-settings-file", "viewsettings.xml",
                "--start", "true",
                "--quit-on-end", "true"]
    traci.start(cmd)

    # GUI演示时：在结果目录下准备截图子目录
    screenshot_dir = None
    if use_gui:
        ts_gui = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join("screenshots", ts_gui)
        os.makedirs(screenshot_dir, exist_ok=True)

    try:
        sim_steps = int(os.getenv("SIM_STEPS", "3600"))
    except Exception:
        sim_steps = 3600
    if sim_steps <= 0:
        sim_steps = 3600
    cooldown_steps = int(np.ceil(LC_COOLDOWN / STEP_LEN))
    lc_cnt = cav_lc = 0
    phase1_lc = 0        # 突发期换道次数
    phase2_lc = 0        # 有序期换道次数
    total_veh        = 0
    collisions_total = 0
    reaction_delay_samples = []   # 记录各车首次感知事故到反应到期的延迟(s)

    # 时序记录（每10步采一次）
    ts_time, ts_queue, ts_speed = [], [], []
    lc_logs = []
    coop_stats = {
        "coop_request_cnt": 0,
        "coop_success_cnt": 0,
        "coop_fail_cnt": 0,
        "platoon_event_cnt": 0,
        "platoon_veh_cnt": 0,
        "coop_response_times": [],
        "coop_gap_build_times": [],
    }
    
    # ★ 创新统计指标

    for step in range(sim_steps):
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            # GUI窗口被用户手动关闭，正常退出本次仿真
            if use_gui:
                print(f"    [GUI] 窗口已关闭，仿真提前结束于 t={step * STEP_LEN:.1f}s")
            break
        t = step * STEP_LEN
        _cur_step = step
        active_ids = set(traci.vehicle.getIDList())
        release_cooperative_yield(active_ids, step, coop_stats)

        # GUI演示时每10步截图一次（约1fps@10Hz步长）
        # 改为 False 避免因为 OpenGL 抢占导致的 GUI 假死/卡在一两秒的情况
        ENABLE_SCREENSHOT = False 
        if use_gui and screenshot_dir and step % 10 == 0 and ENABLE_SCREENSHOT:
            try:
                frame_path = os.path.join(screenshot_dir, f"frame_{step:06d}.png")
                traci.gui.screenshot("View #0", frame_path)
            except Exception:
                pass

        # 动力学与能耗记录
        for v in active_ids:
            cur_acc = traci.vehicle.getAcceleration(v)
            if v in _acc_buf:
                jerk = abs(cur_acc - _acc_buf[v]) / STEP_LEN
                if jerk > MAX_JERK_LIMIT:
                    _perf_metrics["jerk_violations"] += 1
            _acc_buf[v] = cur_acc

            hist = _fol_acc_hist.setdefault(v, [])
            hist.append(cur_acc)
            if len(hist) > FOLLOWER_BEHAV_WINDOW:
                del hist[:-FOLLOWER_BEHAV_WINDOW]
            
            if cur_acc > MAX_LONG_ACC + 0.1 or cur_acc < -MAX_LONG_DEC - 0.1:
                _perf_metrics["acc_violations"] += 1
                
            _perf_metrics["energy_g"] += traci.vehicle.getFuelConsumption(v) * STEP_LEN / 1000.0

        for oid in list(_fol_acc_hist.keys()):
            if oid not in active_ids:
                _fol_acc_hist.pop(oid, None)

        # 三阶段状态推进：准备 -> 执行 -> 稳定
        for v in list(_lc_state.keys()):
            if v not in active_ids:
                del _lc_state[v]
                continue
            state = _lc_state[v]
            phase = state["phase"]
            if phase == "prepare" and step >= state["prep_end"]:
                try:
                    road_id_v = traci.vehicle.getRoadID(v)
                    cur_lane_v = traci.vehicle.getLaneIndex(v)
                    if road_id_v and (not road_id_v.startswith(":")) and cur_lane_v == state["from_lane"]:
                        exec_phase = state.get("decision_phase", "informed")
                        if not lanechange_hard_safety_check(v, road_id_v, state["target_lane"], exec_phase):
                            del _lc_state[v]
                            online_update_weights(success=False, hard_brake=False)
                            continue
                        traci.vehicle.changeLane(v, state["target_lane"], state["exec_dur"])
                        exec_steps = int(np.ceil(state["exec_dur"] / STEP_LEN))
                        state["phase"] = "execute"
                        state["exec_end"] = step + exec_steps
                        mark_cooperative_success(v, step, coop_stats)
                        _last_lc_step[v] = step

                        lc_cnt += 1
                        cav_lc += 1

                        # 改进 #3：互惠记忆 — 检测后车是否减速配合
                        fol_vid = state.get("fol_id", "")
                        if fol_vid and not is_vehicle_cav(fol_vid):
                            try:
                                fol_acc = traci.vehicle.getAcceleration(fol_vid)
                                prev = _human_coop_memory.get(fol_vid, 0.5)
                                if fol_acc < -0.3:
                                    # 后车减速了 → 合作
                                    _human_coop_memory[fol_vid] = min(1.0, prev + 0.20)
                                elif fol_acc > 0.2:
                                    # 后车加速了 → 不合作
                                    _human_coop_memory[fol_vid] = max(0.1, prev - 0.10)
                                # 否则保持不变（维持原速）
                            except traci.exceptions.TraCIException:
                                pass

                        lat_acc = estimate_lat_acc(state["exec_dur"])
                        # 记录此次换道发生时的事故感知阶段
                        veh_pos_v = traci.vehicle.getLanePosition(v)
                        acc_phase = get_vehicle_phase(v, veh_pos_v)
                        if acc_phase == "sudden":
                            phase1_lc += 1
                        elif acc_phase == "informed":
                            phase2_lc += 1
                        lc_logs.append({
                            "label": label,
                            "step": step,
                            "time_s": round(t, 2),
                            "vid": v,
                            "veh_type": "cav",
                            "from_lane": state["from_lane"],
                            "to_lane": state["target_lane"],
                            "lc_stage": "execute",
                            "accident_phase": acc_phase,
                            "prep_s": round(state["prep_dur"], 2),
                            "exec_s": round(state["exec_dur"], 2),
                            "stab_s": round(state["stab_dur"], 2),
                            "lat_acc_est": round(lat_acc, 3),
                            "lat_acc_limit": MAX_LAT_ACC,
                            "within_limit": int(lat_acc <= MAX_LAT_ACC + 1e-6),
                        })
                    else:
                        del _lc_state[v]
                except Exception:
                    del _lc_state[v]
            elif phase == "execute" and step >= state["exec_end"]:
                stab_steps = int(np.ceil(state["stab_dur"] / STEP_LEN))
                state["phase"] = "stabilize"
                state["stab_end"] = step + stab_steps
            elif phase == "stabilize" and step >= state["stab_end"]:
                online_update_weights(success=True, hard_brake=False)
                del _lc_state[v]

        # ── 阶段0→1：正常行驶期结束，尝试从已有车辆中触发动态事故 ──
        if t >= ACCIDENT_TIME and not _accident_state["happened"]:
            if try_create_accident(step):
                _accident_state["happened"]    = True
                _accident_state["time_actual"] = t
                print(f"    [t={t:.1f}s] 事故触发！障碍车辆: {OBSTACLE_IDS}")

        # ── 阶段1→2：广播激活（突发期 → 有序疏散期）──
        if (_accident_state["happened"]
                and not _accident_state["broadcast_active"]
                and t >= _accident_state["time_actual"] + BROADCAST_DELAY):
            _accident_state["broadcast_active"] = True
            print(f"    [t={t:.1f}s] 全局V2X广播激活，进入有序疏散期。")

        manage_obstacles()
        collisions_total += traci.simulation.getCollidingVehiclesNumber()
        obstacle_anchor = get_obstacle_anchor_position(active_ids) if _accident_state["happened"] else ACCIDENT_START
        prepare_admit_cnt = 0

        # 每10步采集时序数据
        if step % 10 == 0:
            ts_time.append(round(t, 1))
            # 事故区后方（900~1000m）队列长度
            queue = 0
            for qvid in traci.vehicle.getIDList():
                if qvid in OBSTACLE_IDS:
                    continue
                try:
                    if traci.vehicle.getRoadID(qvid) != "E0":
                        continue
                    if traci.vehicle.getLaneIndex(qvid) != ACCIDENT_LANE:
                        continue
                    qpos = traci.vehicle.getLanePosition(qvid)
                except traci.exceptions.TraCIException:
                    continue
                if 800 <= qpos <= ACCIDENT_START:
                    queue += 1
            ts_queue.append(queue)
            # 全路段平均速度
            speed_samples = []
            for svid in traci.vehicle.getIDList():
                if svid in OBSTACLE_IDS:
                    continue
                try:
                    speed_samples.append(traci.vehicle.getSpeed(svid))
                except traci.exceptions.TraCIException:
                    continue
            spd  = float(np.mean(speed_samples)) if speed_samples else 0.0
            ts_speed.append(round(spd, 2))

        allowed_vids = None  # 全部允许

        # 换道决策
        for vid in list(traci.vehicle.getIDList()):
            if vid in OBSTACLE_IDS:
                continue
            try:
                road_id = traci.vehicle.getRoadID(vid)
                if not road_id or road_id.startswith(":"):
                    continue
                cur_lane = traci.vehicle.getLaneIndex(vid)
                lane_num = traci.edge.getLaneNumber(road_id)
                veh_pos  = traci.vehicle.getLanePosition(vid)
            except traci.exceptions.TraCIException:
                continue

            # 混合交通：人类车不参与博弈换道和自定义跟驰，由 SUMO SL2015+Krauss 处理
            # SL2015 自带碰撞避免，不需要 setSpeed 干预（会打断 Krauss）
            if not is_vehicle_cav(vid):
                continue

            # 事故区专属限速控制（仅事故发生后启用）
            if _accident_state["happened"]:
                apply_speed_limit_vehicle(vid, veh_pos)
                if apply_emergency_braking_coverage(vid, road_id, cur_lane, veh_pos, obstacle_anchor):
                    continue
            else:
                release_emergency_braking_control(vid)

            # ── 判断当前车辆处于哪个事故感知阶段 ──
            acc_phase = get_vehicle_phase(vid, veh_pos)
            if acc_phase == "normal":
                apply_normal_cruise_control(vid)
                continue
            release_normal_cruise_control(vid)

            # 首次感知事故：分配随机反应延迟，延迟未到期则跳过
            if not check_and_assign_reaction(vid, step, acc_phase, road_id, cur_lane, veh_pos):
                continue

            # 若刚刚解除反应延迟（本步首次可决策），记录延迟时长
            if vid in _reaction_delays and step == _reaction_delays[vid]:
                aware_step = _aware_start_steps.get(vid, step)
                reaction_delay_samples.append((step - aware_step) * STEP_LEN)

            # 决策减频：仅每 DECISION_INTERVAL 步执行博弈，其余步跳过
            if step % DECISION_INTERVAL != 0:
                continue

            if cur_lane != ACCIDENT_LANE:
                continue   # 仅对事故车道上的车辆做换道决策

            if _accident_state["happened"] and road_id == "E0":
                dist_to_obstacle = obstacle_anchor - veh_pos
                if 0.0 < dist_to_obstacle <= EMERGENCY_NO_LC_DIST:
                    continue   # 近障碍末端禁止新发起并线，改为刹停兜底

            detect  = V2X_RANGE if acc_phase == "sudden" else get_accident_broadcast_range(vid)
            trig_s  = ACCIDENT_START - detect
            if not (trig_s < veh_pos < ACCIDENT_END):
                continue

            if vid in _lc_state:
                continue   # 三阶段换道执行中，本步跳过

            if step - _last_lc_step.get(vid, -10**9) < cooldown_steps:
                continue   # 冷却时间内避免频繁换道


            targets = []
            if cur_lane > 0:
                targets.append(cur_lane - 1)
            if lane_num > 2 and cur_lane < lane_num - 1:
                targets.append(cur_lane + 1)

            best_score, best_tgt, best_platoon = -float('inf'), None, [vid]
            for tgt in targets:
                cand_platoon = get_platoon_members(vid, road_id, cur_lane)
                # 高密度与突发期不做多车并行编队换道，降低冲突风险
                if len(cand_platoon) > 1:
                    local_density = estimate_local_density(road_id, cur_lane, veh_pos)
                    if acc_phase == "sudden" or local_density > 0.55:
                        cand_platoon = [vid]
                score = decide_platoon_lanechange(cand_platoon, cur_lane, tgt, road_id, acc_phase)
                if score > -float('inf') and tgt == cur_lane - 1:
                    score += 0.01   # 右侧变道附加微小收益，打破平局
                if score > best_score:
                    best_score, best_tgt, best_platoon = score, tgt, list(cand_platoon)

            if best_tgt is not None and best_score > -float('inf'):
                tgt    = best_tgt
                platoon = list(best_platoon)
                admit_limit = PREP_ADMIT_LIMIT_SUDDEN if acc_phase == "sudden" else PREP_ADMIT_LIMIT_INFORMED
                if prepare_admit_cnt >= admit_limit:
                    continue
                tgt_lid = f"{road_id}_{tgt}"
                lead_id, lead_gap, fol_id, fol_gap = get_neighbors(vid, veh_pos, tgt_lid)
                if not lanechange_hard_safety_check(vid, road_id, tgt, acc_phase):
                    continue
                if apply_cooperative_yield(vid, fol_id, fol_gap):
                    coop_stats["coop_request_cnt"] += 1
                if len(platoon) > 1:
                    coop_stats["platoon_event_cnt"] += 1
                    coop_stats["platoon_veh_cnt"] += len(platoon)
                prep_dur, exec_dur, stab_dur = get_lc_profile(vid)
                prep_steps = int(np.ceil(prep_dur / STEP_LEN))
                for member in platoon:
                    if prepare_admit_cnt >= admit_limit:
                        break
                    if member in _lc_state:
                        continue
                    if step - _last_lc_step.get(member, -10**9) < cooldown_steps:
                        continue
                    if not lanechange_hard_safety_check(member, road_id, tgt, acc_phase):
                        continue
                    _lc_state[member] = {
                        "phase":       "prepare",
                        "from_lane":   cur_lane,
                        "target_lane": tgt,
                        "fol_id":      fol_id,      # 用于互惠记忆检测
                        "decision_phase": acc_phase,
                        "prep_dur":    prep_dur,
                        "exec_dur":    exec_dur,
                        "stab_dur":    stab_dur,
                        "prep_end":    step + prep_steps,
                        "exec_end":    -1,
                        "stab_end":    -1,
                    }
                    prepare_admit_cnt += 1
                    

        total_veh += traci.simulation.getArrivedNumber()

    # 从 tripinfo 读取行程数据（独立文件名，避免多实验冲突）
    tt_total, time_losses = 0.0, []
    tripinfo_count = 0
    if os.path.exists(tripinfo_file):
        for trip in sumolib.xml.parse_fast(tripinfo_file, "tripinfo", ["duration", "timeLoss"]):
            dur = float(trip.duration)
            if dur > 10.0:  # 过滤异常短的数据（<10s 不可能跑完 3.5km）
                tt_total += dur
                time_losses.append(float(trip.timeLoss))
                tripinfo_count += 1
    # 用 tripinfo 的条数覆盖 total_veh，保证分母分子同源
    if tripinfo_count > 0:
        total_veh = tripinfo_count

    try:
        traci.close()
    except Exception:
        pass
    OBSTACLE_IDS.clear()
    _irl_feature_log_snapshot.clear()
    _irl_feature_log_snapshot.extend(_irl_feature_log)
    _irl_feature_log.clear()
    _lc_state.clear()
    _last_lc_step.clear()
    _perc_buf.clear()
    _coop_state.clear()
    _fol_acc_hist.clear()
    _reaction_delays.clear()
    _aware_start_steps.clear()
    _phase_aware_vids.clear()
    _normal_ctrl_vids.clear()
    _emergency_brake_vids.clear()

    # GUI演示结束后尝试用ffmpeg合成视频 (只有当启用截图时才合成)
    ENABLE_SCREENSHOT = False
    if use_gui and screenshot_dir and os.path.isdir(screenshot_dir) and ENABLE_SCREENSHOT:
        frames = [f for f in os.listdir(screenshot_dir) if f.endswith(".png")]
        if frames:
            import subprocess
            video_path = screenshot_dir + ".mp4"
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-framerate", "10",
                "-i", os.path.join(screenshot_dir, "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    视频已保存: {video_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"    截图已保存至: {screenshot_dir}  (共{len(frames)}帧)")
                print(f"    如需合成视频，请安装ffmpeg后在该目录执行:")
                print(f"    ffmpeg -framerate 10 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    avg_coop_response = float(np.mean(coop_stats["coop_response_times"])) if coop_stats["coop_response_times"] else 0.0
    avg_gap_build = float(np.mean(coop_stats["coop_gap_build_times"])) if coop_stats["coop_gap_build_times"] else 0.0
    coop_success_rate = coop_stats["coop_success_cnt"] / max(coop_stats["coop_request_cnt"], 1)

    # ── 综合评价指标：舒适性 + 公平性 ──
    comfort_metrics = metrics_mod.compute_comfort_metrics(
        dict(_fol_acc_hist), STEP_LEN,
        jerk_comfort_threshold=4.0,
    )
    fairness_metrics = metrics_mod.compute_fairness_metrics(
        time_losses,
        [float(t) for t in []] if False else time_losses,  # travel_times ≈ time_losses 正相关
    )
    # 公平性需要真实行程时间，但 tripinfo 只存了 duration 和 timeLoss
    # 用 tripinfo 重新解析获得 per-vehicle travel time
    veh_travel_times = []
    if os.path.exists(tripinfo_file):
        for trip in sumolib.xml.parse_fast(tripinfo_file, "tripinfo", ["duration"]):
            veh_travel_times.append(float(trip.duration))
    fairness_metrics = metrics_mod.compute_fairness_metrics(time_losses, veh_travel_times)

    return {
        "label":           label,
        "profile":         ACTIVE_PROFILE,
        "n_cav":           n_cav,
        "cav_ratio":       1.0,
        "total_vehicles":  total_veh,
        "avg_travel_time": round(tt_total / max(total_veh, 1), 2),
        "avg_delay":       round(float(np.mean(time_losses)) if time_losses else 0.0, 2),
        "lc_cnt":          lc_cnt,
        "cav_lc":          cav_lc,
        "phase1_lc":       phase1_lc,
        "phase2_lc":       phase2_lc,
        "collisions":      collisions_total,
        "max_queue":       max(ts_queue) if ts_queue else 0,
        "accident_time":   round(_accident_state["time_actual"], 1),
        "broadcast_time":  round(_accident_state["time_actual"] + BROADCAST_DELAY, 1)
                           if _accident_state["time_actual"] > 0 else -1.0,
        "avg_reaction_delay": round(float(np.mean(reaction_delay_samples))
                                    if reaction_delay_samples else 0.0, 2),
        "coop_request_cnt": coop_stats["coop_request_cnt"],
        "coop_success_cnt": coop_stats["coop_success_cnt"],
        "coop_fail_cnt":    coop_stats["coop_fail_cnt"],
        "coop_success_rate": round(coop_success_rate, 3),
        "avg_coop_response_s": round(avg_coop_response, 2),
        "avg_gap_build_s": round(avg_gap_build, 2),
        "platoon_event_cnt": coop_stats["platoon_event_cnt"],
        "platoon_veh_cnt": coop_stats["platoon_veh_cnt"],
        "packet_loss_rate": round(_perf_metrics["packet_loss_cnt"] / max(_perf_metrics["comm_msgs"], 1), 3),
        "comm_msgs":       _perf_metrics["comm_msgs"],
        "jerk_violations": _perf_metrics["jerk_violations"],
        "acc_violations":  _perf_metrics["acc_violations"],
        "total_energy_kg": round(_perf_metrics["energy_g"] / 1000.0, 2),
        "ts_time":         ts_time,
        "ts_queue":        ts_queue,
        "ts_speed":        ts_speed,
        "lc_logs":         lc_logs,
        # ★ 创新统计信息
        # ★ 综合评价指标
        **comfort_metrics,
        **fairness_metrics,
    }

# ====================== 批量场景 ======================
def run_simulation():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results   = []
    all_lc_logs = []
    out_dir = os.path.join("results", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # 选择参数标定预设
    try:
        profile_input = input(
            "\n>>> 选择参数集 [b=balanced/bp=balanced_plus/c=conservative/a=aggressive/all] (默认b): "
        ).strip().lower()
    except Exception:
        profile_input = ""

    if profile_input in ("all", "*", "m", "multi", "compare"):
        selected_profiles = ["balanced", "conservative", "aggressive"]
    else:
        selected_profiles = [apply_parameter_profile(profile_input or "balanced")]

    print(f">>> 本次参数集: {', '.join(selected_profiles)}")

    # 场景矩阵：[CAV数, 标签]
    scenarios = [
        (120, "1200pcu/h"),
        (200, "2000pcu/h"),
        (280, "2800pcu/h"),
        (360, "3600pcu/h"),
    ]

    for pidx, profile_name in enumerate(selected_profiles, start=1):
        apply_parameter_profile(profile_name)
        cfg = PROFILE_PRESETS[profile_name]
        print(f"\n===== 参数集 {pidx}/{len(selected_profiles)}: {profile_name} =====")
        print(
            f"    常态时距={cfg['normal_headway']:.2f}s, "
            f"突发换道阈值={cfg['game_min_gain_sudden']:.3f}, "
            f"有序换道阈值={cfg['game_min_gain_informed']:.3f}"
        )

        for idx, (n_cav, lbl) in enumerate(scenarios):
            print(f"\n>>> 场景 {idx+1}/{len(scenarios)} [{profile_name}]: {lbl} ...")
            r = run_once(n_cav, lbl, use_gui=False)  # 数据采集全部 headless
            results.append(r)
            for row in r["lc_logs"]:
                row_copy = dict(row)
                row_copy["profile"] = profile_name
                all_lc_logs.append(row_copy)
            print(f"    通过车辆: {r['total_vehicles']}  平均行程: {r['avg_travel_time']}s  "
                  f"延误: {r['avg_delay']}s  换道: {r['lc_cnt']}次  "
                  f"协同成功率: {r['coop_success_rate']:.2%}  最大队列: {r['max_queue']}辆  碰撞: {r['collisions']}次")
            print(f"    舒适性: Jerk均值={r.get('jerk_mean', 0):.3f}m/s³  P95={r.get('jerk_p95', 0):.3f}  "
                  f"违规率={r.get('jerk_comfort_violation_rate', 0):.2%}")
            print(f"    公平性: 延误Gini={r.get('delay_gini', 0):.4f}  行程Gini={r.get('travel_time_gini', 0):.4f}  "
                  f"延误CV={r.get('delay_cv', 0):.3f}")

    # 保存 CSV
    csv_rows = []
    for r in results:
        csv_rows.append({k: v for k, v in r.items()
                         if k not in ("ts_time", "ts_queue", "ts_speed", "lc_logs")})
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, f"results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存: {csv_path}")

    if len(selected_profiles) > 1 and not df.empty:
        summary_df = (df.groupby("profile", as_index=False)
                        .agg(mean_total_vehicles=("total_vehicles", "mean"),
                             mean_avg_travel_time=("avg_travel_time", "mean"),
                             mean_avg_delay=("avg_delay", "mean"),
                             mean_lc_cnt=("lc_cnt", "mean"),
                             mean_coop_success_rate=("coop_success_rate", "mean"),
                             mean_max_queue=("max_queue", "mean"),
                             mean_jerk_mean=("jerk_mean", "mean"),
                             mean_jerk_p95=("jerk_p95", "mean"),
                             mean_jerk_comfort_violation_rate=("jerk_comfort_violation_rate", "mean"),
                             mean_delay_gini=("delay_gini", "mean"),
                             mean_travel_time_gini=("travel_time_gini", "mean")))
        summary_path = os.path.join(out_dir, f"profile_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"参数集汇总已保存: {summary_path}")

    if all_lc_logs:
        lc_log_path = os.path.join(out_dir, f"lanechange_dynamics_{timestamp}.csv")
        pd.DataFrame(all_lc_logs).to_csv(lc_log_path, index=False, encoding="utf-8-sig")
        print(f"换道动力学日志已保存: {lc_log_path}（共 {len(all_lc_logs)} 条）")

    if len(selected_profiles) == 1:
        generate_plots(results, timestamp, out_dir)
        generate_coop_plots(results, timestamp, out_dir)
        generate_ext_plots(results, timestamp, out_dir)
    else:
        for profile_name in selected_profiles:
            sub_results = [r for r in results if r.get("profile") == profile_name]
            if not sub_results:
                continue
            tag = f"{timestamp}_{profile_name}"
            generate_plots(sub_results, tag, out_dir)
            generate_coop_plots(sub_results, tag, out_dir)
            generate_ext_plots(sub_results, tag, out_dir)
        generate_profile_comparison_plots(results, timestamp, out_dir)

    # 将SUMO输出文件也归档到本次结果目录
    for name in [tripinfo_file, lc_file, fcd_file]:
        if os.path.exists(name):
            try:
                shutil.move(name, os.path.join(out_dir, name))
            except Exception:
                pass

    # 归档本次临时路由文件
    if os.path.exists("tmp_routes.rou.xml"):
        try:
            shutil.move("tmp_routes.rou.xml", os.path.join(out_dir, "tmp_routes.rou.xml"))
        except Exception:
            pass

    # 创建screenshots目录用于GUI演示截图
    screenshots_dir = os.path.join(out_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    # 自动化回归时可通过环境变量跳过GUI演示
    if os.getenv("SKIP_GUI_DEMO", "0").strip().lower() in {"1", "true", "yes", "y"}:
        print(">>> 检测到 SKIP_GUI_DEMO=1，已跳过 GUI 演示。")
        return

    # 所有数据跑完后，询问是否演示所有场景的GUI
    import sys
    try:
        demo_all = input("\n>>> 是否演示所有场景的GUI？(y/n，默认n): ").strip().lower() == 'y'
    except:
        demo_all = False

    demo_profile = selected_profiles[0]
    apply_parameter_profile(demo_profile)
    if len(selected_profiles) > 1:
        print(f">>> 多参数集模式下，GUI演示默认采用参数集: {demo_profile}")
    
    if demo_all:
        print(">>> 依次演示所有场景的GUI...")
        for idx, (n_cav, lbl) in enumerate(scenarios):
            print(f"\n>>> GUI演示 场景 {idx+1}/{len(scenarios)} [{demo_profile}]: {lbl} ...")
            run_once(n_cav, f"GUI演示 {lbl} [{demo_profile}]", use_gui=True)
    else:
        # 默认只演示中等密度场景
        print(f"\n>>> 启动 GUI 演示（2000pcu/h | {demo_profile}）请点击 Start 播放 ...")
        run_once(200, f"GUI演示 2000pcu/h [{demo_profile}]", use_gui=True)

def generate_profile_comparison_plots(results: list, ts: str, out_dir: str):
    """多参数集模式下的跨参数对比图。"""
    if not results:
        return

    rows = []
    for r in results:
        rows.append({
            "profile": r.get("profile", "balanced"),
            "label": r.get("label", ""),
            "total_vehicles": r.get("total_vehicles", 0),
            "avg_delay": r.get("avg_delay", 0.0),
            "coop_success_rate": r.get("coop_success_rate", 0.0),
            "lc_cnt": r.get("lc_cnt", 0),
        })

    df = pd.DataFrame(rows)
    if df.empty or df["profile"].nunique() < 2:
        return

    palette = {
        "balanced": "#1f77b4",
        "balanced_plus": "#17becf",
        "conservative": "#2ca02c",
        "aggressive": "#d62728",
    }
    order = [p for p in ["balanced", "balanced_plus", "conservative", "aggressive"]
             if p in set(df["profile"].tolist())]

    agg = (df.groupby("profile", as_index=False)
             .agg(mean_total_vehicles=("total_vehicles", "mean"),
                  mean_avg_delay=("avg_delay", "mean"),
                  mean_coop_success_rate=("coop_success_rate", "mean"),
                  mean_lc_cnt=("lc_cnt", "mean")))
    agg["profile"] = pd.Categorical(agg["profile"], categories=order, ordered=True)
    agg = agg.sort_values("profile")
    colors = [palette.get(p, "#7f7f7f") for p in agg["profile"]]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].bar(agg["profile"], agg["mean_total_vehicles"], color=colors)
    axes[0].set_title("平均通过车辆")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    axes[1].bar(agg["profile"], agg["mean_avg_delay"], color=colors)
    axes[1].set_title("平均延误（s）")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    axes[2].bar(agg["profile"], agg["mean_coop_success_rate"] * 100.0, color=colors)
    axes[2].set_title("平均协同成功率（%）")
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    axes[3].bar(agg["profile"], agg["mean_lc_cnt"], color=colors)
    axes[3].set_title("平均换道次数")
    axes[3].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    cmp_path = os.path.join(out_dir, f"profile_comparison_{ts}.png")
    plt.savefig(cmp_path, dpi=300, bbox_inches='tight')

    pivot_delay = df.pivot_table(index="label", columns="profile", values="avg_delay", aggfunc="mean")
    pivot_delay = pivot_delay.reindex(columns=order)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    pivot_delay.plot(kind='bar', ax=ax2, color=[palette.get(c, "#7f7f7f") for c in pivot_delay.columns])
    ax2.set_title("不同流量场景下各参数集延误对比")
    ax2.set_ylabel("平均延误（s）")
    ax2.set_xlabel("场景")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    delay_path = os.path.join(out_dir, f"profile_delay_by_scenario_{ts}.png")
    plt.savefig(delay_path, dpi=300, bbox_inches='tight')
    print(f"参数集对比图已保存: {cmp_path}")
    print(f"分场景延误对比图已保存: {delay_path}")

# ====================== 图表 ======================
def generate_plots(results: list, ts: str, out_dir: str, show_plot: bool = False):
    lbl = [r["label"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ---- 图1: 四指标对比柱状图 ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("全CAV 条件下不同交通密度事故路段仿真结果对比", fontsize=14, fontweight='bold')
    metrics = [("total_vehicles", "通过车辆数（辆）"),
               ("avg_travel_time", "平均行程时间（s）"),
               ("avg_delay",       "平均时间损失（s）"),
               ("max_queue",       "最大排队长度（辆）")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(lbl, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.1f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparison_{ts}.png"), dpi=300, bbox_inches='tight')

    # ---- 图2: 事故区队列长度时序 ----
    # 取第一个结果的事故/广播时刻作为标注基准（各场景理论上相同）
    t_accident  = results[0]["accident_time"]  if results else ACCIDENT_TIME
    t_broadcast = results[0]["broadcast_time"] if results else ACCIDENT_TIME + BROADCAST_DELAY
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for r, c in zip(results, colors):
        ax2.plot(r["ts_time"], r["ts_queue"], label=r["label"], color=c, linewidth=1.8)
    ax2.axvline(x=t_accident,  color='red',    linestyle='--', alpha=0.8, label=f'事故发生(t={t_accident:.0f}s)')
    ax2.axvline(x=t_broadcast, color='orange', linestyle=':',  alpha=0.8, label=f'广播激活(t={t_broadcast:.0f}s) 进入有序疏散期')
    ax2.set_xlabel("仿真时间（s）", fontsize=11)
    ax2.set_ylabel("事故区后方队列长度（辆）", fontsize=11)
    ax2.set_title("事故区后方中间车道队列长度随时间变化", fontsize=13)
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"queue_timeseries_{ts}.png"), dpi=300, bbox_inches='tight')

    # ---- 图3: 平均速度时序 ----
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    for r, c in zip(results, colors):
        ax3.plot(r["ts_time"], r["ts_speed"], label=r["label"], color=c, linewidth=1.8)
    ax3.axvline(x=t_accident,  color='red',    linestyle='--', alpha=0.8, label=f'事故发生(t={t_accident:.0f}s)')
    ax3.axvline(x=t_broadcast, color='orange', linestyle=':',  alpha=0.8, label=f'广播激活(t={t_broadcast:.0f}s) 进入有序疏散期')
    ax3.set_xlabel("仿真时间（s）", fontsize=11)
    ax3.set_ylabel("全路段平均速度（m/s）", fontsize=11)
    ax3.set_title("全路段平均行驶速度随时间变化", fontsize=13)
    ax3.legend()
    ax3.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"speed_timeseries_{ts}.png"), dpi=300, bbox_inches='tight')

    # ---- 图4: 两阶段换道次数对比柱状图 ----
    if results:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        x      = np.arange(len(lbl))
        width  = 0.35
        p1vals = [r["phase1_lc"] for r in results]
        p2vals = [r["phase2_lc"] for r in results]
        ax4.bar(x - width/2, p1vals, width=width, label="突发混乱期换道", color="#d62728")
        ax4.bar(x + width/2, p2vals, width=width, label="有序疏散期换道", color="#2ca02c")
        ax4.set_xticks(x)
        ax4.set_xticklabels(lbl, rotation=15)
        ax4.set_title("两阶段换道次数对比（突发期 vs 有序疏散期）", fontsize=12)
        ax4.set_ylabel("换道次数")
        ax4.legend()
        ax4.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"phase_lanechange_{ts}.png"), dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close('all')
    print(f"图表已保存（时间戳: {ts}）")

def generate_coop_plots(results: list, ts: str, out_dir: str):
    lbl = [r["label"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("全 CAV 协同博弈指标对比", fontsize=14, fontweight='bold')

    req_vals = [r["coop_request_cnt"] for r in results]
    suc_vals = [r["coop_success_cnt"] for r in results]
    fail_vals = [r["coop_fail_cnt"] for r in results]
    x = np.arange(len(lbl))
    width = 0.25
    axes[0].bar(x - width, req_vals, width=width, label="协同请求", color="#1f77b4")
    axes[0].bar(x, suc_vals, width=width, label="协同成功", color="#2ca02c")
    axes[0].bar(x + width, fail_vals, width=width, label="协同失败", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbl, rotation=15)
    axes[0].set_title("协同请求/成功/失败次数")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    rate_vals = [r["coop_success_rate"] * 100.0 for r in results]
    axes[1].bar(lbl, rate_vals, color=colors)
    axes[1].set_title("协同成功率（%）")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(rate_vals):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)

    resp_vals = [r["avg_coop_response_s"] for r in results]
    gap_vals = [r["avg_gap_build_s"] for r in results]
    axes[2].plot(lbl, resp_vals, marker='o', label="平均响应时间", color="#ff7f0e")
    axes[2].plot(lbl, gap_vals, marker='s', label="平均开gap时间", color="#9467bd")
    axes[2].set_title("协同响应时延")
    axes[2].set_ylabel("时间（s）")
    axes[2].legend()
    axes[2].grid(linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"coop_metrics_{ts}.png"), dpi=300, bbox_inches='tight')

def generate_ext_plots(results: list, ts: str, out_dir: str):
    lbl = [r["label"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("多智能体博弈系统鲁棒性与能耗评估", fontsize=14, fontweight='bold')

    # 1. 动力学违规次数 (Jerk 与 Acc)
    jerk_vals = [r["jerk_violations"] for r in results]
    acc_vals = [r["acc_violations"] for r in results]
    x = np.arange(len(lbl))
    width = 0.35
    axes[0].bar(x - width/2, jerk_vals, width=width, label="舒适性违规(Jerk>3)", color="#e377c2")
    axes[0].bar(x + width/2, acc_vals, width=width, label="硬约束违规(Acc越界)", color="#8c564b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbl, rotation=15)
    axes[0].set_title("动力学约束违规次数")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. 总能耗评估
    en_vals = [r["total_energy_kg"] for r in results]
    axes[1].bar(lbl, en_vals, color="#17becf")
    axes[1].set_title("系统总能耗/燃油消耗 (kg)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(en_vals):
        axes[1].text(i, v + max(en_vals)*0.02 if max(en_vals)>0 else 0.1, f"{v:.1f}", ha='center', fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)

    # 3. 通信负载与丢包
    msg_vals = [r["comm_msgs"] for r in results]
    loss_vals = [r["packet_loss_rate"] * 100.0 for r in results]
    
    ax3_twin = axes[2].twinx()
    axes[2].bar(lbl, msg_vals, color="#bcbd22", alpha=0.7)
    axes[2].set_ylabel("通信消息总量(次)")
    
    ax3_twin.plot(lbl, loss_vals, color="red", marker='o', linewidth=2)
    ax3_twin.set_ylabel("实际感知丢包率(%)")
    ax3_twin.set_ylim(0, max(loss_vals)*1.5 if max(loss_vals)>0 else 15)

    axes[2].set_title("V2X通信负载与丢包率评估")
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"robustness_metrics_{ts}.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_simulation()
