"""
run_parallel_baseline.py — 多核并行基线对比实验
===============================================
用 multiprocessing 并行执行 5 模型 × 4 场景 = 20 个仿真。

用法:
  python run_parallel_baseline.py                  # 默认 4 进程
  python run_parallel_baseline.py --workers 6       # 6 进程
  python run_parallel_baseline.py --dry-run         # 仅打印任务列表
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from functools import partial

# ─── 导入原有 baseline_comparison 的 running functions ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_comparison as bc

# ====================== 任务定义 ======================

SCENARIOS = [(120, "1200pcu/h"), (200, "2000pcu/h"), (280, "2800pcu/h"), (360, "3600pcu/h")]
MODELS_RUN = {
    "Game (Ours)": bc.run_game,
    "SUMO Default": bc.run_sumo_default,
    "Rule-Based": bc.run_rule_based,
    "No-V2X": bc.run_no_v2x,
    "DRL (PPO)": bc.run_drl,
}


def worker_init():
    """每个 worker 进程启动时的初始化 (预加载 PPO 模型)."""
    # 预加载 PPO 模型到缓存
    if hasattr(bc, '_get_ppo_policy'):
        bc._get_ppo_policy()


def run_one_task(model_name: str, run_func, n_cav: int, label: str,
                 cav_rate: float = 1.0) -> dict:
    """在单个进程中执行一次 (model, scenario) 仿真.
    cav_rate: CAV 渗透率 [0,1]"""
    import game_lane_change as _glc
    _glc.PPO_POLICY = None
    _glc.V2X_CHANNEL = "ideal"
    _glc.CAV_PENETRATION = cav_rate

    # 统一用 "ModelName_Scenario" 作为 label, 确保 tripinfo 文件名唯一
    unique_label = f"{model_name}_{label}"
    # 清理不合法的文件名字符
    safe_model_label = unique_label.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")

    from datetime import datetime
    print(f"  [{model_name}] {label} ...", flush=True)
    t0 = time.time()
    try:
        result = run_func(n_cav, safe_model_label)
        elapsed = time.time() - t0
        print(f"  [{model_name}] {label} OK {elapsed:.0f}s  "
              f"veh:{result.get('total_vehicles',0)} lc:{result.get('lc_cnt',0)}  "
              f"delay:{result.get('avg_delay',0):.1f}s queue:{result.get('max_queue',0)}",
              flush=True)
        return {"model": model_name, "scenario": label, "result": result, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{model_name}] {label} FAIL {elapsed:.0f}s err: {e}", flush=True)
        import traceback
        traceback.print_exc()
        empty = {
            "label": label, "model": model_name, "n_cav": n_cav,
            "total_vehicles": 0, "avg_travel_time": 0, "avg_delay": 0,
            "lc_cnt": 0, "cav_lc": 0, "collisions": 0, "max_queue": 0,
            "ts_time": [], "ts_queue": [], "ts_speed": [],
            "jerk_mean": 0, "jerk_p95": 0, "jerk_comfort_violation_rate": 0,
            "delay_gini": 0, "travel_time_gini": 0, "delay_cv": 0,
        }
        return {"model": model_name, "scenario": label, "result": empty, "error": str(e)}


def build_tasks(cav_rate: float = 1.0):
    """创建所有 (model, scenario) 任务列表.
    cav_rate 被烘焙进偏函数, 传递给每个 worker."""
    from functools import partial
    _worker = partial(run_one_task, cav_rate=cav_rate)
    tasks = []
    for model_name, run_func in MODELS_RUN.items():
        for n_cav, label in SCENARIOS:
            tasks.append((_worker, model_name, run_func, n_cav, label))
    return tasks


def run_parallel(workers: int = 4, dry_run: bool = False, cav_rate: float = 1.0):
    """主入口: 多核并行运行所有对比实验.
    cav_rate: CAV 渗透率 [0,1]"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"baseline_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    tasks = build_tasks(cav_rate)
    n_total = len(tasks)

    print(f"{'='*60}")
    print(f"  多核并行基线对比实验  (CAV={cav_rate*100:.0f}%)")
    print(f"  {n_total} 个仿真任务 ({len(MODELS_RUN)} 模型 × {len(SCENARIOS)} 场景)")
    print(f"  Worker 数: {workers}  (可用核心: {mp.cpu_count()})")
    print(f"  输出目录: {out_dir}")
    print(f"{'='*60}")
    print()

    if dry_run:
        print("  任务列表:")
        for _, model_name, run_func, n_cav, label in tasks:
            print(f"    {model_name:14s} | {label}")
        print(f"\n  共 {n_total} 个任务")
        return

    # ── 并行执行 ──
    t_start = time.time()
    all_results = {name: {} for name in MODELS_RUN}

    with mp.Pool(processes=workers, initializer=worker_init) as pool:
        async_results = []
        for worker_fn, model_name, run_func, n_cav, label in tasks:
            ar = pool.apply_async(worker_fn, (model_name, run_func, n_cav, label))
            async_results.append(ar)

        # 收集结果
        completed = 0
        for ar in async_results:
            data = ar.get()
            completed += 1
            model_name = data["model"]
            scenario = data["scenario"]
            result = data["result"]
            all_results[model_name][scenario] = result

            # 打印进度
            elapsed_total = time.time() - t_start
            pct = completed / n_total * 100
            print(f"\n  [{completed}/{n_total} {pct:.0f}%] {model_name} / {scenario} done "
                  f"({elapsed_total:.0f}s)", flush=True)

    total_time = time.time() - t_start
    print(f"\n  {'='*60}")
    print(f"  全部 {n_total} 个任务完成！总耗时: {total_time:.0f}s")
    print(f"  {'='*60}")

    # ── 保存+绘图 ──
    print(f"\n  保存结果和生成图表...", flush=True)

    # CSV 导出
    csv_rows = []
    for model_name in MODELS_RUN:
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

    import pandas as pd
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, f"baseline_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV 已保存: {csv_path}")

    # 打印结果表
    bc.print_results_table(all_results)

    # 生成图表
    print(f"\n  生成对比图表...", flush=True)
    try:
        bc.plot_baseline_comparison(all_results, timestamp, out_dir)
        print(f"  图表已保存至: {out_dir}")
    except Exception as e:
        print(f"  绘图出错: {e}")
        import traceback; traceback.print_exc()

    # 清理临时文件
    for prefix in ["tripinfo_", "lanechanges_", "fcd_", "tmp_routes_"]:
        for f in os.listdir("."):
            if f.startswith(prefix) and f.endswith((".xml", ".rou.xml")):
                try:
                    os.remove(f)
                except Exception:
                    pass

    print(f"\n  {'='*60}")
    print(f"  实验完成！结果目录: {out_dir}")
    print(f"  {'='*60}")
    return all_results, out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多核并行基线对比实验")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数 (default: 4)")
    parser.add_argument("--cav-rate", type=float, default=1.0,
                        help="CAV 渗透率 0.0~1.0 (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务，不执行")
    args = parser.parse_args()

    run_parallel(workers=args.workers, dry_run=args.dry_run, cav_rate=args.cav_rate)
