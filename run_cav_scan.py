"""
run_cav_scan.py — CAV 渗透率扫描实验
====================================
对 0%, 25%, 50%, 75%, 100% 五档渗透率,
逐档执行 5 模型 × 4 场景 = 20 个仿真任务。

用法:
  python run_cav_scan.py
  python run_cav_scan.py --rates 0 30 60 100
  python run_cav_scan.py --workers 5 --dry-run
"""

import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_parallel_baseline as rpb

# 默认五档渗透率
DEFAULT_RATES = [0, 25, 50, 75, 100]


def run_cav_scan(rates: list = None, workers: int = 4, dry_run: bool = False):
    """逐档跑渗透率扫描."""
    if rates is None:
        rates = DEFAULT_RATES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", f"cav_scan_{timestamp}")
    os.makedirs(root_dir, exist_ok=True)

    total_tasks = len(rates) * len(rpb.MODELS_RUN) * len(rpb.SCENARIOS)
    print(f"{'='*60}")
    print(f"  CAV 渗透率扫描实验")
    print(f"  {len(rates)} 档渗透率 × {len(rpb.MODELS_RUN)} 模型 × {len(rpb.SCENARIOS)} 场景")
    print(f"  = {total_tasks} 个仿真任务")
    print(f"  Worker: {workers}")
    print(f"  输出根目录: {root_dir}")
    print(f"{'='*60}")

    if dry_run:
        for rate in rates:
            print(f"\n  CAV={rate}%:")
            for model_name in rpb.MODELS_RUN:
                for _, lbl in rpb.SCENARIOS:
                    print(f"    {model_name:14s} | {lbl}")
        print(f"\n  共 {total_tasks} 任务")
        return

    all_summaries = {}
    t_global_start = time.time()

    for ri, rate in enumerate(rates):
        print(f"\n{'='*60}")
        print(f"  [{ri+1}/{len(rates)}] CAV 渗透率 = {rate}%")
        print(f"{'='*60}")

        # 为每档创建子目录
        rate_dir = os.path.join(root_dir, f"cav_{rate:03d}")
        os.makedirs(rate_dir, exist_ok=True)

        # 修改并行 runner 中的全局渗透率
        import game_lane_change as glc
        glc.CAV_PENETRATION = rate / 100.0

        # 调用并行 runner 的核心逻辑
        t_rate_start = time.time()
        results, out_dir = rpb.run_parallel(workers=workers, dry_run=False, cav_rate=rate/100.0)
        t_rate = time.time() - t_rate_start

        # 将结果从 run_parallel 的临时目录移到渗透率专属目录
        import shutil
        for f in os.listdir(out_dir):
            shutil.move(os.path.join(out_dir, f),
                       os.path.join(rate_dir, f))

        all_summaries[rate] = {
            "time_s": round(t_rate, 1),
            "dir": rate_dir,
        }
        print(f"\n  CAV={rate}% 完成, 耗时 {t_rate:.0f}s, 结果: {rate_dir}")

    # 汇总
    t_total = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"  渗透率扫描全部完成!")
    print(f"  总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  结果根目录: {root_dir}")
    for rate, info in all_summaries.items():
        print(f"    CAV={rate:3d}%: {info['time_s']:.0f}s → {info['dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAV 渗透率扫描实验")
    parser.add_argument("--rates", type=int, nargs="*", default=None,
                        help=f"渗透率列表 (default: {' '.join(str(r) for r in DEFAULT_RATES)})")
    parser.add_argument("--workers", type=int, default=5,
                        help="并行进程数 (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出任务，不执行")
    args = parser.parse_args()

    run_cav_scan(rates=args.rates, workers=args.workers, dry_run=args.dry_run)
