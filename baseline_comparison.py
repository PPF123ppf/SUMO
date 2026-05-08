"""
baseline_comparison.py
========================
高速公路事故路段 CAV 换道决策模型 — 基线对比实验

比较四种模型：
  1. Game (Ours)   — 基于博弈论的换道决策
  2. SUMO Default  — 使用 SUMO 内置换道模型（SL2015 子车道），不做TraCI干预换道
  3. Rule-Based    — 基于固定 TTC/间隙阈值的规则式换道，无博弈
  4. No-V2X        — 博弈模型，但关闭全局 V2X 广播

使用方法：
  python baseline_comparison.py
"""

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
import metrics as metrics_mod  # 综合评价指标

# ─── SUMO 环境 ───
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
sumo_bin = r'C:\Program Files (x86)\Eclipse\Sumo\bin'
if sumo_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = sumo_bin + os.pathsep + os.environ.get('PATH', '')

# ─── 从 game_lane_change 导入原始博弈模型 ───
import game_lane_change as glc

# ====================== 共用参数 ======================
ACCIDENT_START        = 3000.0
ACCIDENT_END          = 3200.0
ACCIDENT_LANE         = 1
ACCIDENT_TIME         = 90.0
ACCIDENT_SEARCH_WINDOW = 120.0
STEP_LEN              = 0.1
MIN_SAFE_GAP          = 3.0
BROADCAST_DELAY       = 10.0
V2X_RANGE             = 500.0
GLOBAL_V2X_RANGE      = 1200.0
SLOW_ZONE_START       = 2800.0
SLOW_SPEED            = 16.67
LC_COOLDOWN           = 1.5
PREP_ADMIT_LIMIT      = 2
EMERGENCY_REACT_TIME  = 0.60
EMERGENCY_EFF_DEC     = 4.00
EMERGENCY_MARGIN      = 6.0
EMERGENCY_FORCE_BRAKE_DIST = 95.0
EMERGENCY_NO_LC_DIST  = 90.0
EMERGENCY_MAX_ZONE    = 240.0
EMERGENCY_TRIGGER_BUFFER = 5.0
EMERGENCY_PREP_ABORT_DIST = 70.0
# SIM_STEPS 优先从环境变量读取（便于外部控制仿真时长）
try:
    SIM_STEPS = int(os.getenv("SIM_STEPS", "3600"))
except Exception:
    SIM_STEPS = 3600
if SIM_STEPS <= 0:
    SIM_STEPS = 3600


def gen_rou_xml(n_total: int, path: str = "tmp_routes_baseline.rou.xml",
                for_sumo_default: bool = False):
    """按混合交通动态生成路由文件（CAV + 人类车）。

    for_sumo_default=True：所有车用 SUMO SL2015 原生模型（无 CAV/人类区分）。
    for_sumo_default=False：CAV 关掉 SUMO 原生换道由 Python 控制；人类车用 SL2015。
    """
    glc.CAV_PENETRATION  # 确保从 game_lane_change 读取渗透率
    pen = getattr(glc, 'CAV_PENETRATION', 1.0)
    n_cav = int(n_total * pen)
    n_human = n_total - n_cav

    if for_sumo_default:
        # SUMO Default: 全车使用 SL2015，无类型区分
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="cav" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="33.33" guiShape="passenger" color="0,0,255" laneChangeModel="SL2015" lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0" lcKeepRight="1.0" lcAssertive="0.5" lcSigma="0.5"/>
    <route id="r0" edges="E0"/>
    <flow id="f_cav" type="cav" route="r0" begin="0" end="360" number="{n_total}" departSpeed="max" departLane="random"/>
</routes>"""
    else:
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


def get_obstacle_anchor_position(active_ids: set, obstacle_ids: list) -> float:
    positions = []
    for oid in obstacle_ids:
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


def manage_obstacles(obstacle_ids: list):
    active = set(traci.vehicle.getIDList())
    for oid in list(obstacle_ids):
        if oid in active:
            try:
                traci.vehicle.setSpeed(oid, 0.0)
                traci.vehicle.setLaneChangeMode(oid, 0)
                traci.vehicle.setSpeedMode(oid, 0)
            except traci.exceptions.TraCIException:
                pass
        else:
            obstacle_ids.remove(oid)


def try_create_accident(step: int, obstacle_ids: list) -> bool:
    try:
        lane_id = f"E0_{ACCIDENT_LANE}"
        candidates = []
        for vid in traci.lane.getLastStepVehicleIDs(lane_id):
            if vid in obstacle_ids:
                continue
            pos = traci.vehicle.getLanePosition(vid)
            if abs(pos - ACCIDENT_START) <= ACCIDENT_SEARCH_WINDOW:
                candidates.append((pos, vid))
        if not candidates:
            return False
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:2]
        base_pos = selected[0][0]
        for i, (orig_pos, oid) in enumerate(selected):
            try:
                traci.vehicle.setSpeed(oid, 0.0)
                traci.vehicle.setLaneChangeMode(oid, 0)
                traci.vehicle.setSpeedMode(oid, 0)
                target_pos = base_pos - i * 4.6
                traci.vehicle.moveTo(oid, lane_id, target_pos)
            except Exception:
                pass
            obstacle_ids.append(oid)
        return True
    except Exception:
        return False


def get_neighbors(vid: str, ego_pos: float, tgt_lane_id: str):
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


def compute_ttc(gap: float, rel_speed: float) -> float:
    if rel_speed <= 1e-3:
        return float("inf")
    return max(gap, 0.0) / rel_speed


def dynamic_min_gap(ego_speed: float) -> float:
    return max(MIN_SAFE_GAP, 2.0 + 0.35 * max(ego_speed, 0.0))


def compute_stop_distance(speed: float) -> float:
    v = max(float(speed), 0.0)
    dec = max(EMERGENCY_EFF_DEC, 1e-3)
    return v * EMERGENCY_REACT_TIME + (v * v) / (2.0 * dec) + EMERGENCY_MARGIN


def compute_safe_speed_by_distance(distance: float) -> float:
    dec = max(EMERGENCY_EFF_DEC, 1e-3)
    remain = max(float(distance) - EMERGENCY_MARGIN, 0.0)
    return float(np.sqrt(max(2.0 * dec * remain, 0.0)))


def apply_emergency_braking_coverage(vid, road_id, lane_idx, veh_pos, obstacle_anchor):
    if road_id != "E0" or lane_idx != ACCIDENT_LANE:
        return False
    dist = float(obstacle_anchor - veh_pos)
    if dist <= 0.0 or dist > EMERGENCY_MAX_ZONE:
        return False
    try:
        cur_speed = traci.vehicle.getSpeed(vid)
    except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
        return False
    except Exception:
        return False
    stop_need = compute_stop_distance(cur_speed)
    force_cover = (dist <= EMERGENCY_FORCE_BRAKE_DIST)
    if (not force_cover) and dist > stop_need + EMERGENCY_TRIGGER_BUFFER:
        return False
    target_speed = min(cur_speed, compute_safe_speed_by_distance(dist))
    if force_cover:
        force_cap = max(0.0, 0.12 * max(dist - EMERGENCY_MARGIN, 0.0))
        target_speed = min(target_speed, force_cap)
    if dist <= EMERGENCY_MARGIN + 1.0:
        target_speed = 0.0
    try:
        traci.vehicle.setSpeed(vid, float(np.clip(target_speed, 0.0, max(cur_speed, 0.0))))
    except traci.exceptions.TraCIException:
        return False
    return True


def apply_speed_limit_vehicle(vid: str, veh_pos: float):
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


def collect_ts_data(step, obstacle_ids):
    """采集时序数据"""
    t = step * STEP_LEN
    queue = 0
    for qvid in traci.vehicle.getIDList():
        if qvid in obstacle_ids:
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
    speed_samples = []
    for svid in traci.vehicle.getIDList():
        if svid in obstacle_ids:
            continue
        try:
            speed_samples.append(traci.vehicle.getSpeed(svid))
        except traci.exceptions.TraCIException:
            continue
    spd = float(np.mean(speed_samples)) if speed_samples else 0.0
    return t, queue, round(spd, 2)


def read_tripinfo(filepath: str = "tripinfo.xml"):
    tt_total, time_losses, trip_count = 0.0, [], 0
    if os.path.exists(filepath):
        for trip in sumolib.xml.parse_fast(filepath, "tripinfo", ["duration", "timeLoss"]):
            dur = float(trip.duration)
            if dur > 10.0:  # 过滤异常短的数据
                tt_total += dur
                time_losses.append(float(trip.timeLoss))
                trip_count += 1
    return tt_total, time_losses, trip_count


def read_travel_times(filepath: str = "tripinfo.xml"):
    """从 tripinfo 读取各车行程时间。"""
    times = []
    if os.path.exists(filepath):
        for trip in sumolib.xml.parse_fast(filepath, "tripinfo", ["duration"]):
            times.append(float(trip.duration))
    return times


# ====================== 五大模型 ======================

def run_game(n_cav: int, label: str) -> dict:
    """模型1: 原博弈模型（同时博弈）"""
    return glc.run_once(n_cav, label)


def run_sumo_default(n_cav: int, label: str) -> dict:
    """
    模型2: SUMO 默认换道模型
    ─────────────────────────
    - 使用SUMO原生跟驰(Krauss) + 换道(LC2013)
    - 不通过TraCI干预车辆的换道决策
    - 仅保留：事故创建、障碍管理、限速区、紧急制动兜底
    """
    obstacle_ids: list = []
    accident_state = {"happened": False, "time_actual": -1.0, "broadcast_active": False}
    lc_cnt = 0

    safe_lbl = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    tripinfo_f = f"tripinfo_{safe_lbl}.xml"
    rou_file = gen_rou_xml(n_cav, for_sumo_default=True)
    cmd = [r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo', "-c", "accident_highway.sumocfg",
           "--route-files", rou_file,
           "--lateral-resolution", "0.8",
           "--collision.action", "warn",
           "--no-warnings", "true",
           "--no-step-log", "true",
           "--tripinfo-output", tripinfo_f,
           "--start", "true",
           "--quit-on-end", "true"]
    traci.start(cmd)

    total_veh = collisions_total = 0
    ts_time, ts_queue, ts_speed = [], [], []
    acc_buf: dict = {}
    fol_acc_hist: dict = {}

    for step in range(SIM_STEPS):
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            break
        t = step * STEP_LEN

        # 加速度 / Jerk 记录
        for v in traci.vehicle.getIDList():
            if v in obstacle_ids:
                continue
            try:
                cur_acc = traci.vehicle.getAcceleration(v)
            except traci.exceptions.TraCIException:
                continue
            hist = fol_acc_hist.setdefault(v, [])
            hist.append(cur_acc)
            if len(hist) > 15:
                del hist[:-15]

        if t >= ACCIDENT_TIME and not accident_state["happened"]:
            if try_create_accident(step, obstacle_ids):
                accident_state["happened"] = True
                accident_state["time_actual"] = t

        if (accident_state["happened"] and not accident_state["broadcast_active"]
                and t >= accident_state["time_actual"] + BROADCAST_DELAY):
            accident_state["broadcast_active"] = True

        manage_obstacles(obstacle_ids)
        collisions_total += traci.simulation.getCollidingVehiclesNumber()
        active_ids = set(traci.vehicle.getIDList())
        obstacle_anchor = (get_obstacle_anchor_position(active_ids, obstacle_ids)
                           if accident_state["happened"] else ACCIDENT_START)

        if step % 10 == 0:
            tt, q, sp = collect_ts_data(step, obstacle_ids)
            ts_time.append(tt)
            ts_queue.append(q)
            ts_speed.append(sp)

        # 限速和紧急制动（安全兜底，不影响SUMO原生换道逻辑）
        for vid in traci.vehicle.getIDList():
            if vid in obstacle_ids:
                continue
            try:
                road_id = traci.vehicle.getRoadID(vid)
                if not road_id or road_id.startswith(":"):
                    continue
                cur_lane = traci.vehicle.getLaneIndex(vid)
                veh_pos = traci.vehicle.getLanePosition(vid)
            except traci.exceptions.TraCIException:
                continue
            if accident_state["happened"]:
                apply_speed_limit_vehicle(vid, veh_pos)
                apply_emergency_braking_coverage(vid, road_id, cur_lane, veh_pos, obstacle_anchor)

        total_veh += traci.simulation.getArrivedNumber()

    tt_total, time_losses, trip_cnt = read_tripinfo(tripinfo_f)
    if trip_cnt > 0:
        total_veh = trip_cnt
    veh_travel_times = read_travel_times(tripinfo_f)
    try:
        traci.close()
    except Exception:
        pass

    comfort = metrics_mod.compute_comfort_metrics(dict(fol_acc_hist), STEP_LEN)
    fairness = metrics_mod.compute_fairness_metrics(time_losses, veh_travel_times)

    return {
        "label": label, "model": "SUMO Default", "n_cav": n_cav,
        "total_vehicles": total_veh,
        "avg_travel_time": round(tt_total / max(total_veh, 1), 2),
        "avg_delay": round(float(np.mean(time_losses)) if time_losses else 0.0, 2),
        "lc_cnt": lc_cnt, "collisions": collisions_total,
        "max_queue": max(ts_queue) if ts_queue else 0,
        "ts_time": ts_time, "ts_queue": ts_queue, "ts_speed": ts_speed,
        **comfort, **fairness,
    }


def run_rule_based(n_cav: int, label: str) -> dict:
    """
    模型3: 规则式换道（Rule-Based Baseline）
    ──────────────────────────────────────
    - 无博弈论、无协同让行、无编队换道
    - 固定 TTC 阈值 3.0s 触发换道
    - 基于感知模型（延迟+噪声+丢包）
    - 紧急制动兜底保留
    """
    obstacle_ids: list = []
    accident_state = {"happened": False, "time_actual": -1.0, "broadcast_active": False}
    _last_lc_step: dict = {}
    cooldown_steps = int(np.ceil(LC_COOLDOWN / STEP_LEN))
    RULE_TTC_THRESHOLD = 3.0
    lc_cnt = 0

    safe_lbl = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    tripinfo_f = f"tripinfo_{safe_lbl}.xml"
    rou_file = gen_rou_xml(n_cav, for_sumo_default=False)
    cmd = [r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo', "-c", "accident_highway.sumocfg",
           "--route-files", rou_file,
           "--no-warnings", "true",
           "--no-step-log", "true",
           "--tripinfo-output", tripinfo_f,
           "--lanechange.duration", "1.0",
           "--collision.action", "warn",
           "--start", "true",
           "--quit-on-end", "true"]
    traci.start(cmd)

    total_veh = collisions_total = 0
    ts_time, ts_queue, ts_speed = [], [], []
    acc_buf: dict = {}
    fol_acc_hist: dict = {}

    for step in range(SIM_STEPS):
        try:
            traci.simulationStep()
        except traci.exceptions.FatalTraCIError:
            break
        t = step * STEP_LEN

        # 加速度 / Jerk 记录
        for v in traci.vehicle.getIDList():
            if v in obstacle_ids:
                continue
            try:
                cur_acc = traci.vehicle.getAcceleration(v)
            except traci.exceptions.TraCIException:
                continue
            hist = fol_acc_hist.setdefault(v, [])
            hist.append(cur_acc)
            if len(hist) > 15:
                del hist[:-15]

        if t >= ACCIDENT_TIME and not accident_state["happened"]:
            if try_create_accident(step, obstacle_ids):
                accident_state["happened"] = True
                accident_state["time_actual"] = t

        if (accident_state["happened"] and not accident_state["broadcast_active"]
                and t >= accident_state["time_actual"] + BROADCAST_DELAY):
            accident_state["broadcast_active"] = True

        manage_obstacles(obstacle_ids)
        collisions_total += traci.simulation.getCollidingVehiclesNumber()
        active_ids = set(traci.vehicle.getIDList())
        obstacle_anchor = (get_obstacle_anchor_position(active_ids, obstacle_ids)
                           if accident_state["happened"] else ACCIDENT_START)

        if step % 10 == 0:
            tt, q, sp = collect_ts_data(step, obstacle_ids)
            ts_time.append(tt)
            ts_queue.append(q)
            ts_speed.append(sp)

        # ── 换道决策（逐车） ──
        for vid in traci.vehicle.getIDList():
            if vid in obstacle_ids:
                continue
            try:
                road_id = traci.vehicle.getRoadID(vid)
                if not road_id or road_id.startswith(":"):
                    continue
                cur_lane = traci.vehicle.getLaneIndex(vid)
                lane_num = traci.edge.getLaneNumber(road_id)
                veh_pos = traci.vehicle.getLanePosition(vid)
            except traci.exceptions.TraCIException:
                continue

            if accident_state["happened"]:
                apply_speed_limit_vehicle(vid, veh_pos)
                if apply_emergency_braking_coverage(vid, road_id, cur_lane, veh_pos, obstacle_anchor):
                    continue

            # 混合交通：人类车不参与规则换道，由 SUMO 原生处理
            if not glc.is_vehicle_cav(vid):
                continue

            if cur_lane != ACCIDENT_LANE:
                continue
            acc_phase = glc.get_vehicle_phase(vid, veh_pos)
            if acc_phase == "normal":
                continue
            if accident_state["happened"] and road_id == "E0":
                dist_to_obstacle = obstacle_anchor - veh_pos
                if 0.0 < dist_to_obstacle <= EMERGENCY_NO_LC_DIST:
                    continue
            if step - _last_lc_step.get(vid, -10**9) < cooldown_steps:
                continue

            ego_spd = traci.vehicle.getSpeed(vid)
            targets = []
            if cur_lane > 0:
                targets.append(cur_lane - 1)
            if lane_num > 2 and cur_lane < lane_num - 1:
                targets.append(cur_lane + 1)

            best_tgt = None
            for tgt in targets:
                tgt_lid = f"{road_id}_{tgt}"
                lead_id, lead_gap, fol_id, fol_gap = glc.sample_perception(vid, veh_pos, tgt_lid)
                dyn_gap = dynamic_min_gap(ego_spd)
                if lead_gap < dyn_gap or fol_gap < dyn_gap:
                    continue
                lead_spd = glc.perc_speed(lead_id) if lead_id else ego_spd
                fol_spd = glc.perc_speed(fol_id) if fol_id else 0.0
                ttc_front = compute_ttc(lead_gap, max(ego_spd - lead_spd, 0.0))
                ttc_rear = compute_ttc(fol_gap, max(fol_spd - ego_spd, 0.0))
                if min(ttc_front, ttc_rear) >= RULE_TTC_THRESHOLD:
                    best_tgt = tgt
                    break

            if best_tgt is not None and glc.lanechange_hard_safety_check(vid, road_id, best_tgt, acc_phase):
                traci.vehicle.changeLane(vid, best_tgt, 2.8)
                _last_lc_step[vid] = step
                lc_cnt += 1

        total_veh += traci.simulation.getArrivedNumber()

    tt_total, time_losses, trip_cnt = read_tripinfo(tripinfo_f)
    if trip_cnt > 0:
        total_veh = trip_cnt
    veh_travel_times = read_travel_times(tripinfo_f)
    try:
        traci.close()
    except Exception:
        pass

    comfort = metrics_mod.compute_comfort_metrics(dict(fol_acc_hist), STEP_LEN)
    fairness = metrics_mod.compute_fairness_metrics(time_losses, veh_travel_times)

    return {
        "label": label, "model": "Rule-Based", "n_cav": n_cav,
        "total_vehicles": total_veh,
        "avg_travel_time": round(tt_total / max(total_veh, 1), 2),
        "avg_delay": round(float(np.mean(time_losses)) if time_losses else 0.0, 2),
        "lc_cnt": lc_cnt, "collisions": collisions_total,
        "max_queue": max(ts_queue) if ts_queue else 0,
        "ts_time": ts_time, "ts_queue": ts_queue, "ts_speed": ts_speed,
        **comfort, **fairness,
    }


def run_no_v2x(n_cav: int, label: str) -> dict:
    """模型4: 无V2X广播（全局感知范围=300m）"""
    orig_global = glc.GLOBAL_V2X_RANGE
    orig_delay = glc.BROADCAST_DELAY
    glc.GLOBAL_V2X_RANGE = glc.V2X_RANGE
    try:
        result = glc.run_once(n_cav, label)
        result["model"] = "No-V2X"
        return result
    finally:
        glc.GLOBAL_V2X_RANGE = orig_global
        glc.BROADCAST_DELAY = orig_delay


# ====================== 可视化 ======================

def plot_baseline_comparison(all_results: dict, ts: str, out_dir: str):
    model_colors = {
        "Game (Ours)": "#1f77b4",
        "SUMO Default": "#ff7f0e",
        "Rule-Based": "#2ca02c",
        "No-V2X": "#d62728",
    }
    model_order = ["Game (Ours)", "SUMO Default", "Rule-Based", "No-V2X"]
    scenarios = ["1200pcu/h", "2000pcu/h", "2800pcu/h", "3600pcu/h"]

    # 图1: 6指标分组柱状图
    metrics = [
        ("total_vehicles", "通过车辆数（辆）", False),
        ("avg_travel_time", "平均行程时间（s）", False),
        ("avg_delay", "平均时间损失（s）", False),
        ("max_queue", "最大排队长度（辆）", False),
        ("collisions", "碰撞次数", False),
        ("lc_cnt", "换道次数", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("五种换道模型对比", fontsize=16, fontweight='bold')
    for idx, (key, title, show_legend) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        x = np.arange(len(scenarios))
        width = 0.18
        for mi, model_name in enumerate(model_order):
            vals = [all_results.get(model_name, {}).get(sc, {}).get(key, 0) for sc in scenarios]
            bars = ax.bar(x + (mi - 1.5) * width, vals, width * 0.85,
                         label=model_name, color=model_colors[model_name])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02 if max(vals) > 0 else 0.5,
                            f"{v:.1f}", ha='center', fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if show_legend:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"baseline_comparison_{ts}.png"), dpi=300, bbox_inches='tight')

    # 图2: 队列长度时序
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10))
    fig2.suptitle("事故区后方队列长度时序对比", fontsize=14, fontweight='bold')
    for si, sc in enumerate(scenarios):
        ax = axes2[si // 2][si % 2]
        for model_name in model_order:
            r = all_results.get(model_name, {}).get(sc, {})
            ax.plot(r.get("ts_time", []), r.get("ts_queue", []),
                   label=model_name, color=model_colors[model_name], linewidth=1.5)
        ax.set_title(sc, fontsize=11)
        ax.set_xlabel("时间（s）")
        ax.set_ylabel("队列长度（辆）")
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)
        ax.axvline(x=ACCIDENT_TIME, color='red', linestyle='--', alpha=0.6)
        ax.axvline(x=ACCIDENT_TIME + BROADCAST_DELAY, color='orange', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"baseline_queue_{ts}.png"), dpi=300, bbox_inches='tight')

    # 图3: 速度时序
    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10))
    fig3.suptitle("全路段平均速度时序对比", fontsize=14, fontweight='bold')
    for si, sc in enumerate(scenarios):
        ax = axes3[si // 2][si % 2]
        for model_name in model_order:
            r = all_results.get(model_name, {}).get(sc, {})
            ax.plot(r.get("ts_time", []), r.get("ts_speed", []),
                   label=model_name, color=model_colors[model_name], linewidth=1.5)
        ax.set_title(sc, fontsize=11)
        ax.set_xlabel("时间（s）")
        ax.set_ylabel("平均速度（m/s）")
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)
        ax.axvline(x=ACCIDENT_TIME, color='red', linestyle='--', alpha=0.6)
        ax.axvline(x=ACCIDENT_TIME + BROADCAST_DELAY, color='orange', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"baseline_speed_{ts}.png"), dpi=300, bbox_inches='tight')

    # 图4: 3600高密度场景深度对比
    if "3600pcu/h" in all_results.get("Game (Ours)", {}):
        fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
        fig4.suptitle("高密度场景（3600pcu/h）关键指标对比", fontsize=14, fontweight='bold')
        for ai, (key, title) in enumerate([("avg_delay", "平均延误（s）"),
                                            ("max_queue", "最大队列（辆）"),
                                            ("collisions", "碰撞次数")]):
            ax = axes4[ai]
            vals = [all_results.get(m, {}).get("3600pcu/h", {}).get(key, 0) for m in model_order]
            bars = ax.bar(model_order, vals, color=[model_colors[m] for m in model_order])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                        f"{v:.1f}", ha='center', fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.tick_params(axis='x', rotation=15)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"baseline_high_density_{ts}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')


def print_results_table(all_results: dict):
    model_order = ["Game (Ours)", "SUMO Default", "Rule-Based", "No-V2X"]
    scenarios = ["1200pcu/h", "2000pcu/h", "2800pcu/h", "3600pcu/h"]
    header = (f"{'模型':<14} {'场景':<10} {'通过':<7} {'行程':<8} {'延误':<7} "
              f"{'换道':<6} {'队列':<6} {'碰撞':<5} {'JerkP95':<8} {'违例率':<7} {'Gini':<6}")
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for model_name in model_order:
        first = True
        for sc in scenarios:
            if sc in all_results.get(model_name, {}):
                r = all_results[model_name][sc]
                prefix = model_name if first else ""
                jerk_p95 = r.get('jerk_p95', '-')
                if isinstance(jerk_p95, (int, float)):
                    jerk_p95 = f"{jerk_p95:.3f}"
                viol_rate = r.get('jerk_comfort_violation_rate', '-')
                if isinstance(viol_rate, (int, float)):
                    viol_rate = f"{viol_rate:.2%}"
                gini = r.get('delay_gini', '-')
                if isinstance(gini, (int, float)):
                    gini = f"{gini:.3f}"
                print(f"{prefix:<14} {sc:<10} {r.get('total_vehicles', 0):<7} "
                      f"{r.get('avg_travel_time', 0):<8} {r.get('avg_delay', 0):<7} "
                      f"{r.get('lc_cnt', 0):<6} {r.get('max_queue', 0):<6} "
                      f"{r.get('collisions', 0):<5} {jerk_p95:<8} {viol_rate:<7} {gini:<6}")
                first = False


def run_baseline_comparison():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"baseline_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = [(120, "1200pcu/h"), (200, "2000pcu/h"), (280, "2800pcu/h"), (360, "3600pcu/h")]
    models = {
        "Game (Ours)": run_game,
        "SUMO Default": run_sumo_default,
        "Rule-Based": run_rule_based,
        "No-V2X": run_no_v2x,
    }
    all_results = {name: {} for name in models}

    for model_name, run_func in models.items():
        print(f"\n{'='*60}")
        print(f"  运行模型: {model_name}")
        print(f"{'='*60}")
        for idx, (n_cav, lbl) in enumerate(scenarios):
            print(f"\n  >>> {idx+1}/4 [{model_name}]: {lbl} ...", flush=True)
            try:
                r = run_func(n_cav, lbl)
                all_results[model_name][lbl] = r
                print(f"      通过:{r['total_vehicles']} 行程:{r['avg_travel_time']}s  "
                      f"延误:{r['avg_delay']}s 换道:{r['lc_cnt']}次  "
                      f"队列:{r['max_queue']}辆 碰撞:{r['collisions']}次", flush=True)
            except Exception as e:
                print(f"      [错误] {model_name}/{lbl}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                all_results[model_name][lbl] = {"label": lbl, "model": model_name, "n_cav": n_cav,
                    "total_vehicles": 0, "avg_travel_time": 0, "avg_delay": 0,
                    "lc_cnt": 0, "collisions": 0, "max_queue": 0,
                    "ts_time": [], "ts_queue": [], "ts_speed": [],
                    "jerk_mean": 0, "jerk_p95": 0, "jerk_comfort_violation_rate": 0,
                    "delay_gini": 0, "travel_time_gini": 0, "delay_cv": 0}

    # CSV
    csv_rows = []
    for model_name in models:
        for sc, r in all_results[model_name].items():
            csv_rows.append({"model": model_name, "scenario": sc,
                "total_vehicles": r.get("total_vehicles", 0),
                "avg_travel_time": r.get("avg_travel_time", 0),
                "avg_delay": r.get("avg_delay", 0),
                "lc_cnt": r.get("lc_cnt", 0),
                "collisions": r.get("collisions", 0),
                "max_queue": r.get("max_queue", 0),
                "jerk_mean": r.get("jerk_mean", 0),
                "jerk_p95": r.get("jerk_p95", 0),
                "jerk_comfort_violation_rate": r.get("jerk_comfort_violation_rate", 0),
                "delay_gini": r.get("delay_gini", 0),
                "travel_time_gini": r.get("travel_time_gini", 0),
                "delay_cv": r.get("delay_cv", 0)})
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, f"baseline_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV已保存: {csv_path}")

    print_results_table(all_results)

    print("\n生成对比图表...", flush=True)
    try:
        plot_baseline_comparison(all_results, timestamp, out_dir)
    except Exception as e:
        print(f"  绘图出错: {e}")
        import traceback; traceback.print_exc()

    for name in ["tripinfo.xml", "lanechanges.xml", "fcd.xml", "tmp_routes_baseline.rou.xml", "tmp_routes.rou.xml"]:
        if os.path.exists(name):
            try:
                shutil.move(name, os.path.join(out_dir, name))
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"  基线对比实验完成！结果目录: {out_dir}")
    print(f"{'='*60}")
    return all_results


if __name__ == "__main__":
    run_baseline_comparison()
