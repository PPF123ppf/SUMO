"""
run_cav_scan.py — CAV 渗透率扫描实验（全并行 + 多轮取平均）
=============================================================
所有任务一次扔进池子，workers 空闲即领取，不等档。
支持 --reps N 多轮重复以消除随机性。

用法:
  python run_cav_scan.py                          # 单轮 120 任务
  python run_cav_scan.py --reps 3                 # 三轮取平均
  python run_cav_scan.py --rates 0 10 30 50 70 100
  python run_cav_scan.py --workers 20 --dry-run
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_parallel_baseline as rpb
import baseline_comparison as bc

DEFAULT_RATES = [0, 10, 30, 50, 70, 100]


def _worker_init():
    """每个 worker 启动时预加载 PPO."""
    if hasattr(bc, '_get_ppo_policy'):
        bc._get_ppo_policy()


def run_cav_scan(rates: list = None, workers: int = 20,
                 dry_run: bool = False, reps: int = 1):
    if rates is None:
        rates = DEFAULT_RATES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", f"cav_scan_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    single_count = len(rates) * len(rpb.MODELS_RUN) * len(rpb.SCENARIOS)
    n_total = single_count * reps
    print(f"{'='*60}")
    print(f"  CAV 渗透率扫描 {'(多轮取平均)' if reps > 1 else ''}")
    print(f"  {len(rates)} 档 × {len(rpb.MODELS_RUN)} 模型 × {len(rpb.SCENARIOS)} 场景")
    if reps > 1:
        print(f"  {reps} 轮重复 = {n_total} 任务")
    else:
        print(f"  = {n_total} 任务")
    print(f"  Worker: {workers}  (可用核: {mp.cpu_count()})")
    print(f"  输出: {root_dir}")
    print(f"{'='*60}\n")

    if dry_run:
        for rate in rates:
            for model_name in rpb.MODELS_RUN:
                for _, lbl in rpb.SCENARIOS:
                    print(f"  CAV={rate:3d}% | {model_name:14s} | {lbl}"
                          + (f" ×{reps}" if reps > 1 else ""))
        print(f"\n  共 {n_total} 任务")
        return

    # ── 构建所有任务 ──
    tasks = []  # (worker_fn, model_name, run_func, n_cav, label, rate, seed)
    for rep in range(reps):
        seed = rep  # 每轮固定种子 0,1,2,...
        for rate in rates:
            worker_fn = partial(rpb.run_one_task, cav_rate=rate / 100.0, seed=seed)
            for model_name, run_func in rpb.MODELS_RUN.items():
                for n_cav, label in rpb.SCENARIOS:
                    label_full = f"CAV{rate:03d}_{model_name}_{label}_s{seed}"
                    safe = label_full.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")
                    tasks.append((worker_fn, model_name, run_func, n_cav, safe, rate, seed))

    # ── 全并行执行 ──
    t_start = time.time()
    # 按 (rate, model, scenario, seed) 存储原始结果
    raw_results = {}
    completed = 0

    with mp.Pool(processes=workers, initializer=_worker_init) as pool:
        async_results = []
        for (worker_fn, model_name, run_func, n_cav, label_full, rate, seed) in tasks:
            ar = pool.apply_async(worker_fn, (model_name, run_func, n_cav, label_full))
            async_results.append(ar)

        for ar in async_results:
            data = ar.get()
            completed += 1
            label_raw = data["scenario"]
            # 解析: "CAV030_Game_Ours_1200pcu_h_s0"
            parts = label_raw.split("_s")
            seed_val = int(parts[-1]) if len(parts) > 1 else 0
            scenario_part = parts[0]
            rate_str = scenario_part.split("_", 1)[0].replace("CAV", "")
            rate_val = int(rate_str)
            model_name = data["model"]
            result = data["result"]

            key = (rate_val, model_name, scenario_part)
            if key not in raw_results:
                raw_results[key] = []
            raw_results[key].append(result)

            elapsed_total = time.time() - t_start
            pct = completed / n_total * 100
            print(f"\n  [{completed}/{n_total} {pct:.0f}%] CAV={rate_val}% {model_name} "
                  f"s={seed_val} veh:{result.get('total_vehicles',0)} lc:{result.get('lc_cnt',0)} "
                  f"({elapsed_total:.0f}s)", flush=True)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部 {n_total} 任务完成！总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"{'='*60}")

    # ── 保存原始数据 + 平均值 ──
    import pandas as pd
    import numpy as np

    raw_rows = []
    avg_rows = []
    for (rate_val, model_name, scenario_part), results_list in raw_results.items():
        # 取场景名（去掉 CAVxxx_ 前缀）
        sc_clean = scenario_part.split("_", 1)[-1] if "_" in scenario_part else scenario_part

        # 原始数据行 (每个 seed 一行)
        for r in results_list:
            raw_rows.append({
                "cav_rate": rate_val, "model": model_name, "scenario": sc_clean,
                "total_vehicles": r.get("total_vehicles", 0),
                "avg_travel_time": r.get("avg_travel_time", 0),
                "avg_delay": r.get("avg_delay", 0),
                "lc_cnt": r.get("lc_cnt", 0),
                "collisions": r.get("collisions", 0),
                "max_queue": r.get("max_queue", 0),
                "jerk_p95": r.get("jerk_p95", 0),
                "jerk_comfort_violation_rate": r.get("jerk_comfort_violation_rate", 0),
                "delay_gini": r.get("delay_gini", 0),
            })

        # 平均值行 (多轮时)
        if reps > 1:
            def _avg(key):
                vals = [r.get(key, 0) for r in results_list]
                return round(np.mean(vals), 1)
            avg_rows.append({
                "cav_rate": rate_val, "model": model_name, "scenario": sc_clean,
                "total_vehicles": _avg("total_vehicles"),
                "avg_travel_time": _avg("avg_travel_time"),
                "avg_delay": _avg("avg_delay"),
                "lc_cnt": _avg("lc_cnt"),
                "collisions": round(np.mean([r.get("collisions", 0) for r in results_list]), 1),
                "max_queue": _avg("max_queue"),
                "jerk_p95": _avg("jerk_p95"),
                "jerk_comfort_violation_rate": _avg("jerk_comfort_violation_rate"),
                "delay_gini": _avg("delay_gini"),
            })

    df_raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(root_dir, f"cav_scan_raw_{timestamp}.csv")
    df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"  原始数据: {raw_path}")

    if avg_rows:
        df_avg = pd.DataFrame(avg_rows)
        avg_path = os.path.join(root_dir, f"cav_scan_avg_{timestamp}.csv")
        df_avg.to_csv(avg_path, index=False, encoding="utf-8-sig")
        print(f"  平均值:  {avg_path}")

    # ── 打印各档平均结果表 ──
    for rate in rates:
        print(f"\n── CAV={rate}% {'(平均)' if reps > 1 else ''} ──")
        table_data = {name: {} for name in rpb.MODELS_RUN}
        for model_name in rpb.MODELS_RUN:
            for sc in ["1200pcu/h", "2000pcu/h", "2800pcu/h", "3600pcu/h"]:
                # 找对应结果
                candidates = [r for r in (avg_rows if reps > 1 else raw_rows)
                             if r["cav_rate"] == rate and r["model"] == model_name and sc in r["scenario"]]
                if candidates:
                    r = candidates[0]
                    table_data[model_name][sc] = {
                        "total_vehicles": r["total_vehicles"],
                        "avg_travel_time": r["avg_travel_time"],
                        "avg_delay": r["avg_delay"],
                        "lc_cnt": r["lc_cnt"],
                        "collisions": r["collisions"],
                        "max_queue": r["max_queue"],
                        "jerk_p95": r["jerk_p95"],
                        "jerk_comfort_violation_rate": r["jerk_comfort_violation_rate"],
                        "delay_gini": r["delay_gini"],
                    }
        bc.print_results_table(table_data)

    print(f"\n{'='*60}")
    print(f"  完成！结果目录: {root_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAV 渗透率扫描（全并行 + 多轮）")
    parser.add_argument("--rates", type=int, nargs="*", default=None,
                        help=f"渗透率列表 (default: {' '.join(str(r) for r in DEFAULT_RATES)})")
    parser.add_argument("--workers", type=int, default=20,
                        help="并行进程数 (default: 20)")
    parser.add_argument("--reps", type=int, default=1,
                        help="每任务重复次数 (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务")
    args = parser.parse_args()
    run_cav_scan(rates=args.rates, workers=args.workers,
                 dry_run=args.dry_run, reps=args.reps)
