"""
run_cav_scan.py — CAV 渗透率扫描实验（全并行版）
=================================================
所有渗透率任务一次扔进池子，workers 空闲即领取，不等档。

用法:
  python run_cav_scan.py
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


def run_cav_scan(rates: list = None, workers: int = 20, dry_run: bool = False):
    if rates is None:
        rates = DEFAULT_RATES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", f"cav_scan_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    n_total = len(rates) * len(rpb.MODELS_RUN) * len(rpb.SCENARIOS)
    print(f"{'='*60}")
    print(f"  CAV 渗透率扫描 (全并行)")
    print(f"  {len(rates)} 档 × {len(rpb.MODELS_RUN)} 模型 × {len(rpb.SCENARIOS)} 场景 = {n_total} 任务")
    print(f"  Worker: {workers}  (可用核: {mp.cpu_count()})")
    print(f"  输出: {root_dir}")
    print(f"{'='*60}\n")

    if dry_run:
        for rate in rates:
            for model_name in rpb.MODELS_RUN:
                for _, lbl in rpb.SCENARIOS:
                    print(f"  CAV={rate:3d}% | {model_name:14s} | {lbl}")
        print(f"\n  共 {n_total} 任务")
        return

    # ── 构建所有任务 ──
    tasks = []  # (worker_fn, model_name, run_func, n_cav, label_with_rate)
    for rate in rates:
        worker_fn = partial(rpb.run_one_task, cav_rate=rate / 100.0)
        for model_name, run_func in rpb.MODELS_RUN.items():
            for n_cav, label in rpb.SCENARIOS:
                # label 含 rate 和 model 前缀，确保 tripinfo 唯一
                label_full = f"CAV{rate:03d}_{model_name}_{label}"
                safe = label_full.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("(", "").replace(")", "")
                tasks.append((worker_fn, model_name, run_func, n_cav, safe, rate))

    # ── 全并行执行 ──
    t_start = time.time()
    # 按 rate 分组的结果
    all_results = {}
    for rate in rates:
        all_results[rate] = {name: {} for name in rpb.MODELS_RUN}
    completed = 0

    with mp.Pool(processes=workers, initializer=_worker_init) as pool:
        async_results = []
        for worker_fn, model_name, run_func, n_cav, label_full, rate in tasks:
            ar = pool.apply_async(worker_fn, (model_name, run_func, n_cav, label_full))
            async_results.append(ar)

        for ar in async_results:
            data = ar.get()
            completed += 1
            # 从 label 解析出 rate
            label_raw = data["scenario"]
            # label_raw 形如 "CAV030_Game_Ours_1200pcu_h"
            parts = label_raw.split("_", 1)
            rate_str = parts[0].replace("CAV", "")
            rate_val = int(rate_str)
            model_name = data["model"]
            scenario_label = data["scenario"]  # 存完整的
            result = data["result"]
            all_results[rate_val][model_name][scenario_label] = result

            elapsed_total = time.time() - t_start
            pct = completed / n_total * 100
            print(f"\n  [{completed}/{n_total} {pct:.0f}%] CAV={rate_val}% {model_name} "
                  f"veh:{result.get('total_vehicles',0)} lc:{result.get('lc_cnt',0)} "
                  f"({elapsed_total:.0f}s)", flush=True)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部 {n_total} 任务完成！总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"{'='*60}")

    # ── 保存 CSV ──
    import pandas as pd
    all_csv_rows = []
    for rate in rates:
        for model_name in rpb.MODELS_RUN:
            for sc, r in all_results[rate][model_name].items():
                all_csv_rows.append({
                    "cav_rate": rate,
                    "model": model_name,
                    "scenario": sc,
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
    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(root_dir, f"cav_scan_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {csv_path}")

    # ── 打印各档结果表 ──
    for rate in rates:
        print(f"\n── CAV={rate}% ──")
        # 构建兼容 print_results_table 的 dict
        table_data = {name: {} for name in rpb.MODELS_RUN}
        for model_name in rpb.MODELS_RUN:
            for sc, r in all_results[rate][model_name].items():
                # 提取场景名（去掉 CAVxxx_ModelName_ 前缀）
                sc_clean = sc.split("_", 2)[-1] if "_" in sc else sc
                table_data[model_name][sc_clean] = r
        bc.print_results_table(table_data)

    print(f"\n{'='*60}")
    print(f"  完成！结果目录: {root_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAV 渗透率扫描（全并行）")
    parser.add_argument("--rates", type=int, nargs="*", default=None,
                        help=f"渗透率列表 (default: {' '.join(str(r) for r in DEFAULT_RATES)})")
    parser.add_argument("--workers", type=int, default=20,
                        help="并行进程数 (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务")
    args = parser.parse_args()
    run_cav_scan(rates=args.rates, workers=args.workers, dry_run=args.dry_run)
