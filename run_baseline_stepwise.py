"""
run_baseline_stepwise.py
========================
分步运行基线对比实验（支持断点续跑、增量保存）

使用方法：
  完整运行（每轮约3分钟×16轮=48分钟）：
    python run_baseline_stepwise.py

  快速验证（SIM_STEPS=600 即60秒仿真，~5分钟出结果）：
    set SIM_STEPS=600 && python run_baseline_stepwise.py

  只跑部分模型/场景：
    python run_baseline_stepwise.py --models "Game (Ours),SUMO Default"
    python run_baseline_stepwise.py --scenarios "1200pcu/h,2000pcu/h"

  继续上次中断的运行：
    python run_baseline_stepwise.py --resume results\baseline_20260424_160000
"""

import os
import sys
import json
import shutil
import argparse
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# ─── 确保基线模块可导入 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# ====================== 配置 ======================
SCENARIOS = [
    (120, "1200pcu/h"),
    (200, "2000pcu/h"),
    (280, "2800pcu/h"),
    (360, "3600pcu/h"),
]

PENETRATIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

MODELS = {
    "Game (Ours)":   "run_game",
    "SUMO Default":  "run_sumo_default",
    "Rule-Based":    "run_rule_based",
    "No-V2X":        "run_no_v2x",
}

# ====================== 辅助函数 ======================

def load_checkpoint(out_dir: str):
    """从 checkpoint 文件恢复已完成的轮次"""
    ckpt_path = os.path.join(out_dir, "_checkpoint.json")
    if not os.path.exists(ckpt_path):
        return set()
    try:
        with open(ckpt_path, "r") as f:
            done = set(tuple(item) for item in json.load(f))
        return done
    except Exception:
        return set()


def save_checkpoint(out_dir: str, done: set):
    """保存已完成的轮次到 checkpoint 文件"""
    ckpt_path = os.path.join(out_dir, "_checkpoint.json")
    try:
        with open(ckpt_path, "w") as f:
            json.dump(sorted(list(done)), f)
    except Exception as e:
        print(f"  [警告] 保存 checkpoint 失败: {e}")


def save_intermediate_results(out_dir: str, all_results: dict, timestamp: str):
    """增量保存 CSV 和时序数据"""
    # 1. 概要指标 CSV
    csv_rows = []
    for model_name, scen_dict in all_results.items():
        for sc, r in scen_dict.items():
            csv_rows.append({
                "model": model_name, "scenario": sc,
                "penetration": r.get("penetration", 0),
                "total_vehicles": r.get("total_vehicles", 0),
                "avg_travel_time": r.get("avg_travel_time", 0),
                "avg_delay": r.get("avg_delay", 0),
                "lc_cnt": r.get("lc_cnt", 0),
                "collisions": r.get("collisions", 0),
                "max_queue": r.get("max_queue", 0),
            })
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, f"baseline_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] 概要CSV -> {csv_path}")

    # 2. 时序数据（Pickle，包含ts_time/ts_queue/ts_speed）
    ts_path = os.path.join(out_dir, f"baseline_ts_{timestamp}.pkl")
    ts_data = {}
    for model_name, scen_dict in all_results.items():
        for sc, r in scen_dict.items():
            ts_data[f"{model_name}|{sc}"] = {
                "ts_time":  r.get("ts_time", []),
                "ts_queue": r.get("ts_queue", []),
                "ts_speed": r.get("ts_speed", []),
            }
    with open(ts_path, "wb") as f:
        pickle.dump(ts_data, f)
    print(f"  [保存] 时序数据 -> {ts_path}")

    # 3. 已完成的轮次明细
    detail_rows = []
    for model_name, scen_dict in all_results.items():
        for sc, r in scen_dict.items():
            detail_rows.append({
                "model": model_name, "scenario": sc,
                "penetration": r.get("penetration", 0),
                "n_cav": r.get("n_cav", 0),
                "total_vehicles": r.get("total_vehicles", 0),
                "avg_travel_time": r.get("avg_travel_time", 0),
                "avg_delay": r.get("avg_delay", 0),
                "lc_cnt": r.get("lc_cnt", 0),
                "collisions": r.get("collisions", 0),
                "max_queue": r.get("max_queue", 0),
                "label": r.get("label", ""),
            })
    detail_path = os.path.join(out_dir, f"baseline_detail_{timestamp}.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")


def cleanup_after_round():
    """清理本轮产生的中间文件（Windows 上重试等待文件解锁）"""
    import time as _time
    for f in ["tripinfo.xml", "lanechanges.xml", "fcd.xml",
              "tmp_routes_baseline.rou.xml", "tmp_routes.rou.xml"]:
        for _ in range(10):
            try:
                os.remove(f)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                _time.sleep(0.5)


def run_single(model_name: str, n_cav: int, lbl: str, sim_steps: int,
               penetration: float = 1.0) -> dict:
    """运行单轮实验"""
    from baseline_comparison import run_game, run_sumo_default, run_rule_based, run_no_v2x
    import game_lane_change as glc

    # 设置当前渗透率
    glc.CAV_PENETRATION = penetration

    func_map = {
        "Game (Ours)":  run_game,
        "SUMO Default": run_sumo_default,
        "Rule-Based":   run_rule_based,
        "No-V2X":       run_no_v2x,
    }
    run_func = func_map.get(model_name)
    if run_func is None:
        raise ValueError(f"未知模型: {model_name}")

    os.environ['SIM_STEPS'] = str(sim_steps)
    result = run_func(n_cav, lbl)
    result["penetration"] = penetration
    return result


# ====================== 主入口 ======================

def run_baseline_stepwise():
    parser = argparse.ArgumentParser(description="分步运行基线对比实验")
    parser.add_argument("--models", type=str, default=None,
                        help="要运行的模型列表，逗号分隔，默认全部")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="要运行的场景列表，逗号分隔，默认全部")
    parser.add_argument("--resume", type=str, default=None,
                        help="从已有结果目录恢复运行")
    parser.add_argument("--sim-steps", type=int, default=3600,
                        help="每轮仿真步数（默认3600=360秒）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录（默认自动创建）")
    parser.add_argument("--no-plot", action="store_true",
                        help="全部完成后不绘图")
    parser.add_argument("--penetrations", type=str, default=None,
                        help="CAV渗透率列表，逗号分隔，默认 0.0,0.1,0.3,0.5,0.7,1.0")
    parser.add_argument("--irl-weights", type=str, default=None,
                        help="IRL权重文件路径（如 irl_weights_v2.npz）")
    args = parser.parse_args()

    # 解析模型列表
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        for m in model_names:
            if m not in MODELS:
                print(f"[错误] 未知模型: {m}，可选: {list(MODELS.keys())}")
                sys.exit(1)
    else:
        model_names = list(MODELS.keys())

    # 解析场景列表
    if args.scenarios:
        scen_list = [s.strip() for s in args.scenarios.split(",")]
        valid_scens = [s for _, s in SCENARIOS]
        for s in scen_list:
            if s not in valid_scens:
                print(f"[错误] 未知场景: {s}，可选: {valid_scens}")
                sys.exit(1)
        scens = [(n, l) for n, l in SCENARIOS if l in scen_list]
    else:
        scens = SCENARIOS

    # 解析渗透率列表
    if args.penetrations:
        pen_list = [float(p.strip()) for p in args.penetrations.split(",")]
        pens = pen_list
    else:
        pens = PENETRATIONS

    # 输出目录
    if args.resume:
        out_dir = args.resume
        if not os.path.exists(out_dir):
            print(f"[错误] 续跑目录不存在: {out_dir}")
            sys.exit(1)
        print(f"[续跑] 从已有目录继续: {out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_dir or os.path.join("results", f"baseline_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[新实验] 输出目录: {out_dir}")

    # 加载 checkpoint（已完成的轮次）
    done = load_checkpoint(out_dir) if args.resume else set()
    if done:
        print(f"[续跑] 已发现 {len(done)} 个已完成轮次")

    # 加载 IRL 权重
    if args.irl_weights:
        import numpy as np
        import game_lane_change as glc
        data = np.load(args.irl_weights)
        glc.PAYOFF_WEIGHTS['informed'] = data['informed']
        glc.PAYOFF_WEIGHTS['sudden'] = data['informed'].copy() * 0.8
        print(f"[IRL权重] 已加载: {args.irl_weights}")
        print(f"  informed: {[f'{w:.4f}' for w in glc.PAYOFF_WEIGHTS['informed']]}")

    SIM_STEPS = args.sim_steps
    total_rounds = len(model_names) * len(scens) * len(pens)
    print(f"\n{'='*60}")
    print(f"  分步基线对比实验（混合交通）")
    print(f"  模型: {model_names}")
    print(f"  场景: {[l for _, l in scens]}")
    print(f"  渗透率: {pens}")
    print(f"  总轮数: {total_rounds}  SIM_STEPS={SIM_STEPS}")
    print(f"  {'[续跑模式]' if args.resume else '[全新模式]'}")
    print(f"{'='*60}\n")

    # 恢复已有结果
    all_results = {}
    if args.resume:
        # 尝试从 detail CSV 恢复（取最新的，即行数最多的）
        best_detail_path = None
        best_detail_rows = 0
        for fname in os.listdir(out_dir):
            if fname.startswith("baseline_detail_") and fname.endswith(".csv"):
                detail_path = os.path.join(out_dir, fname)
                try:
                    df = pd.read_csv(detail_path)
                    if len(df) > best_detail_rows:
                        best_detail_rows = len(df)
                        best_detail_path = detail_path
                except Exception as e:
                    print(f"  [警告] 无法加载 {detail_path}: {e}")
        if best_detail_path:
            df = pd.read_csv(best_detail_path)
            for _, row in df.iterrows():
                mn = row["model"]
                sc = row["scenario"]
                if mn not in all_results:
                    all_results[mn] = {}
                pen_val = float(row.get("penetration", 0))
                all_results[mn][sc] = {
                    "label": row.get("label", sc),
                    "model": mn,
                    "penetration": pen_val,
                    "n_cav": int(row["n_cav"]),
                    "total_vehicles": int(row["total_vehicles"]),
                    "avg_travel_time": row["avg_travel_time"],
                    "avg_delay": row["avg_delay"],
                    "lc_cnt": int(row["lc_cnt"]),
                    "collisions": int(row["collisions"]),
                    "max_queue": int(row["max_queue"]),
                    "ts_time": [], "ts_queue": [], "ts_speed": [],
                }
            print(f"  [恢复] 从 {best_detail_path} 加载了 {len(df)} 条记录")

        # 尝试从 pickle 恢复时序数据
        for fname in os.listdir(out_dir):
            if fname.startswith("baseline_ts_") and fname.endswith(".pkl"):
                ts_path = os.path.join(out_dir, fname)
                try:
                    with open(ts_path, "rb") as f:
                        ts_data = pickle.load(f)
                    for key, ts_vals in ts_data.items():
                        mn, sc = key.split("|", 1)
                        if mn in all_results and sc in all_results[mn]:
                            all_results[mn][sc]["ts_time"]  = ts_vals["ts_time"]
                            all_results[mn][sc]["ts_queue"] = ts_vals["ts_queue"]
                            all_results[mn][sc]["ts_speed"] = ts_vals["ts_speed"]
                    print(f"  [恢复] 从 {ts_path} 加载了时序数据")
                except Exception as e:
                    print(f"  [警告] 无法加载时序数据 {ts_path}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 逐轮运行 ──
    round_idx = 0
    for model_name in model_names:
        if model_name not in all_results:
            all_results[model_name] = {}

        for pen in pens:
            pen_label = f"{int(pen * 100)}%CAV"

            for n_cav, lbl in scens:
                round_idx += 1
                full_lbl = f"{lbl}_{pen_label}"
                key = (model_name, full_lbl)

                # 检查是否已完成
                if key in done and full_lbl in all_results.get(model_name, {}):
                    r = all_results[model_name][full_lbl]
                    print(f"  [{round_idx}/{total_rounds}] [OK] 跳过（已完成）: {model_name} / {full_lbl}  "
                          f"(通过{r.get('total_vehicles', 0)} 行程{r.get('avg_travel_time', 0)}s)")
                    continue

                print(f"  [{round_idx}/{total_rounds}] => 运行: {model_name} / {full_lbl}  "
                      f"(n_cav={n_cav}, steps={SIM_STEPS}, pen={pen})", flush=True)
                print(f"      开始时间: {datetime.now().strftime('%H:%M:%S')}")

                # 清理中间文件
                cleanup_after_round()

                try:
                    r = run_single(model_name, n_cav, full_lbl, SIM_STEPS, penetration=pen)
                    all_results[model_name][full_lbl] = r
                    print(f"      + 完成！通过:{r['total_vehicles']} 行程:{r['avg_travel_time']}s  "
                          f"延误:{r['avg_delay']}s 换道:{r['lc_cnt']}次  "
                          f"队列:{r['max_queue']}辆 碰撞:{r['collisions']}次", flush=True)

                    # 标记完成并增量保存
                    done.add(key)
                    save_checkpoint(out_dir, done)
                    save_intermediate_results(out_dir, all_results, timestamp)

                except Exception as e:
                    print(f"      [错误] {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # 创建占位记录
                    all_results[model_name][full_lbl] = {
                        "label": full_lbl, "model": model_name, "n_cav": n_cav,
                        "penetration": pen,
                        "total_vehicles": 0, "avg_travel_time": 0, "avg_delay": 0,
                        "lc_cnt": 0, "collisions": 0, "max_queue": 0,
                        "ts_time": [], "ts_queue": [], "ts_speed": [],
                    }
                    done.add(key)
                    save_checkpoint(out_dir, done)
                    save_intermediate_results(out_dir, all_results, timestamp)

                finally:
                    # 确保 SUMO 连接关闭
                    try:
                        import traci
                        traci.close()
                    except Exception:
                        pass

            end_time = datetime.now().strftime('%H:%M:%S')
            print(f"      结束时间: {end_time}\n")

    # ── 全部完成后生成图表 ──
    if not args.no_plot:
        print(f"\n{'='*60}")
        print("  全部轮次完成！生成对比图表...")
        print(f"{'='*60}\n")
        try:
            from baseline_comparison import plot_baseline_comparison, print_results_table

            print_results_table(all_results)
            plot_baseline_comparison(all_results, timestamp, out_dir)
            print(f"\n  图表已保存至: {out_dir}")
        except Exception as e:
            print(f"  绘图出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  [跳过绘图] --no-plot 已指定")

    # 移动中间文件
    for name in ["tripinfo.xml", "lanechanges.xml", "fcd.xml",
                 "tmp_routes_baseline.rou.xml", "tmp_routes.rou.xml"]:
        if os.path.exists(name):
            try:
                shutil.move(name, os.path.join(out_dir, name))
            except Exception:
                pass

    # 删除 checkpoint 文件
    ckpt_path = os.path.join(out_dir, "_checkpoint.json")
    try:
        os.remove(ckpt_path)
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  基线对比实验完成！")
    print(f"  结果目录: {out_dir}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    run_baseline_stepwise()
