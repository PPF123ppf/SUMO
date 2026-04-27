"""
plot_baseline_results.py
=========================
从已保存的基线对比结果生成可视化图表（无需重跑仿真）

使用方法：
  # 从结果目录自动读取数据并绘图
  python plot_baseline_results.py results\baseline_20260424_154645

  # 指定输出目录
  python plot_baseline_results.py results\baseline_20260424_154645 --out-dir my_charts

  # 查看可用的结果目录
  python plot_baseline_results.py --list
"""

import os
import sys
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


# ====================== 参数（与 baseline_comparison.py 保持一致） ======================
ACCIDENT_TIME      = 90.0
ACCIDENT_END       = 3200.0
ACCIDENT_START     = 3000.0
BROADCAST_DELAY    = 10.0

MODEL_COLORS = {
    "Game (Ours)":  "#1f77b4",
    "SUMO Default": "#ff7f0e",
    "Rule-Based":   "#2ca02c",
    "No-V2X":       "#d62728",
    "LKSQ (Ours)":  "#9467bd",
}
MODEL_ORDER = ["Game (Ours)", "SUMO Default", "Rule-Based", "No-V2X", "LKSQ (Ours)"]
SCENARIOS = ["1200pcu/h", "2000pcu/h", "2800pcu/h", "3600pcu/h"]


# ====================== 数据加载 ======================

def find_result_dirs(base_dir: str = "results"):
    """查找所有基线对比结果目录"""
    pattern = os.path.join(base_dir, "baseline_*")
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d)]


def load_results_from_dir(result_dir: str) -> dict:
    """
    从结果目录加载数据，返回 all_results dict
    格式: {model_name: {scenario: {指标数据, ts_time, ts_queue, ts_speed}}}
    """
    all_results = {}

    # 1. 尝试加载时序数据 pickle（最完整）
    pkl_files = glob.glob(os.path.join(result_dir, "baseline_ts_*.pkl"))
    if pkl_files:
        pkl_path = sorted(pkl_files)[-1]
        try:
            with open(pkl_path, "rb") as f:
                ts_data = pickle.load(f)
            print(f"  [加载] 时序数据: {os.path.basename(pkl_path)}")
            for key, ts_vals in ts_data.items():
                mn, sc = key.split("|", 1)
                if mn not in all_results:
                    all_results[mn] = {}
                if sc not in all_results[mn]:
                    all_results[mn][sc] = {}
                all_results[mn][sc]["ts_time"]  = ts_vals.get("ts_time", [])
                all_results[mn][sc]["ts_queue"] = ts_vals.get("ts_queue", [])
                all_results[mn][sc]["ts_speed"] = ts_vals.get("ts_speed", [])
        except Exception as e:
            print(f"  [警告] 加载时序 pickle 失败: {e}")

    # 2. 加载 detail CSV（包含概要指标）
    detail_files = glob.glob(os.path.join(result_dir, "baseline_detail_*.csv"))
    if not detail_files:
        detail_files = glob.glob(os.path.join(result_dir, "baseline_results_*.csv"))
    
    if detail_files:
        detail_path = sorted(detail_files)[-1]
        try:
            df = pd.read_csv(detail_path)
            print(f"  [加载] 概要指标: {os.path.basename(detail_path)} ({len(df)} 条记录)")
            for _, row in df.iterrows():
                mn = row["model"]
                sc = row["scenario"]
                if mn not in all_results:
                    all_results[mn] = {}
                if sc not in all_results[mn]:
                    all_results[mn][sc] = {}
                all_results[mn][sc].update({
                    "label": row.get("label", sc),
                    "model": mn,
                    "n_cav": int(row["n_cav"]),
                    "total_vehicles": int(row["total_vehicles"]),
                    "avg_travel_time": row["avg_travel_time"],
                    "avg_delay": row["avg_delay"],
                    "lc_cnt": int(row.get("lc_cnt", 0)),
                    "collisions": int(row.get("collisions", 0)),
                    "max_queue": int(row.get("max_queue", 0)),
                })
                # 确保时序字段存在
                if "ts_time" not in all_results[mn][sc]:
                    all_results[mn][sc]["ts_time"] = []
                    all_results[mn][sc]["ts_queue"] = []
                    all_results[mn][sc]["ts_speed"] = []
        except Exception as e:
            print(f"  [警告] 加载 CSV 失败: {e}")

    if not all_results:
        print(f"  [错误] 在 {result_dir} 中未找到可用的结果数据")
        return None

    return all_results


# ====================== 绘图函数 ======================

def plot_comparison_bar(all_results: dict, out_dir: str, ts: str):
    """图1: 6指标分组柱状图"""
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
        x = np.arange(len(SCENARIOS))
        width = 0.18
        for mi, model_name in enumerate(MODEL_ORDER):
            vals = [all_results.get(model_name, {}).get(sc, {}).get(key, 0) for sc in SCENARIOS]
            bars = ax.bar(x + (mi - 1.5) * width, vals, width * 0.85,
                         label=model_name, color=MODEL_COLORS[model_name])
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + max(vals)*0.02 if max(vals) > 0 else 0.5,
                            f"{v:.1f}", ha='center', fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if show_legend:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"baseline_comparison_{ts}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [图表] {path}")
    plt.close(fig)


def plot_queue_timeseries(all_results: dict, out_dir: str, ts: str):
    """图2: 队列长度时序"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("事故区后方队列长度时序对比", fontsize=14, fontweight='bold')
    
    for si, sc in enumerate(SCENARIOS):
        ax = axes[si // 2][si % 2]
        for model_name in MODEL_ORDER:
            r = all_results.get(model_name, {}).get(sc, {})
            ts_time = r.get("ts_time", [])
            ts_queue = r.get("ts_queue", [])
            if ts_time and ts_queue:
                ax.plot(ts_time, ts_queue,
                       label=model_name, color=MODEL_COLORS[model_name], linewidth=1.5)
        ax.set_title(sc, fontsize=11)
        ax.set_xlabel("时间（s）")
        ax.set_ylabel("队列长度（辆）")
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)
        ax.axvline(x=ACCIDENT_TIME, color='red', linestyle='--', alpha=0.6, label='事故')
        ax.axvline(x=ACCIDENT_TIME + BROADCAST_DELAY, color='orange', linestyle=':', alpha=0.6, label='广播')
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"baseline_queue_{ts}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [图表] {path}")
    plt.close(fig)


def plot_speed_timeseries(all_results: dict, out_dir: str, ts: str):
    """图3: 速度时序"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("全路段平均速度时序对比", fontsize=14, fontweight='bold')
    
    for si, sc in enumerate(SCENARIOS):
        ax = axes[si // 2][si % 2]
        for model_name in MODEL_ORDER:
            r = all_results.get(model_name, {}).get(sc, {})
            ts_time = r.get("ts_time", [])
            ts_speed = r.get("ts_speed", [])
            if ts_time and ts_speed:
                ax.plot(ts_time, ts_speed,
                       label=model_name, color=MODEL_COLORS[model_name], linewidth=1.5)
        ax.set_title(sc, fontsize=11)
        ax.set_xlabel("时间（s）")
        ax.set_ylabel("平均速度（m/s）")
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)
        ax.axvline(x=ACCIDENT_TIME, color='red', linestyle='--', alpha=0.6)
        ax.axvline(x=ACCIDENT_TIME + BROADCAST_DELAY, color='orange', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"baseline_speed_{ts}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [图表] {path}")
    plt.close(fig)


def plot_high_density(all_results: dict, out_dir: str, ts: str):
    """图4: 高密度场景深度对比"""
    if "3600pcu/h" not in all_results.get("Game (Ours)", {}):
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("高密度场景（3600pcu/h）关键指标对比", fontsize=14, fontweight='bold')
    
    for ai, (key, title) in enumerate([("avg_delay", "平均延误（s）"),
                                        ("max_queue", "最大队列（辆）"),
                                        ("collisions", "碰撞次数")]):
        ax = axes[ai]
        vals = [all_results.get(m, {}).get("3600pcu/h", {}).get(key, 0) for m in MODEL_ORDER]
        bars = ax.bar(MODEL_ORDER, vals, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.1f}", ha='center', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"baseline_high_density_{ts}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [图表] {path}")
    plt.close(fig)


def plot_radar(all_results: dict, out_dir: str, ts: str):
    """图5: 雷达图综合对比（归一化多维度）"""
    metrics_config = {
        "total_vehicles":  ("通过车辆数", True),   # 越大越好
        "avg_delay":       ("平均延误", False),     # 越小越好
        "max_queue":       ("最大队列", False),     # 越小越好
        "collisions":      ("碰撞次数", False),     # 越小越好
        "lc_cnt":          ("换道次数", None),      # 中性
    }
    
    n_metrics = len(metrics_config)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    for sc in SCENARIOS:
        # 归一化
        vals = {}
        for key, (_, higher_better) in metrics_config.items():
            raw = [all_results.get(m, {}).get(sc, {}).get(key, 0) for m in MODEL_ORDER]
            if max(raw) == min(raw):
                norm = [0.5] * len(raw)
            elif higher_better:
                norm = [(v - min(raw)) / max(max(raw) - min(raw), 1e-6) for v in raw]
            else:
                norm = [(max(raw) - v) / max(max(raw) - min(raw), 1e-6) for v in raw]
            vals[key] = norm
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.suptitle(f"综合性能雷达图 — {sc}", fontsize=14, fontweight='bold')
        
        for mi, model_name in enumerate(MODEL_ORDER):
            values = [vals[key][mi] for key in metrics_config]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=MODEL_COLORS[model_name])
            ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[model_name])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([v[0] for v in metrics_config.values()], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        safe_sc = sc.replace("/", "_").replace("\\", "_")
        path = os.path.join(out_dir, f"baseline_radar_{safe_sc}_{ts}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  [图表] {path}")
        plt.close(fig)


def plot_heatmap(all_results: dict, out_dir: str, ts: str):
    """图6: 模型×场景热力图"""
    for key, title in [("avg_delay", "平均延误（s）"), 
                        ("collisions", "碰撞次数"),
                        ("max_queue", "最大队列（辆）")]:
        data = np.zeros((len(MODEL_ORDER), len(SCENARIOS)))
        for mi, mn in enumerate(MODEL_ORDER):
            for si, sc in enumerate(SCENARIOS):
                data[mi, si] = all_results.get(mn, {}).get(sc, {}).get(key, 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(SCENARIOS)))
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(SCENARIOS)
        ax.set_yticklabels(MODEL_ORDER)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for mi in range(len(MODEL_ORDER)):
            for si in range(len(SCENARIOS)):
                ax.text(si, mi, f"{data[mi, si]:.1f}", ha="center", va="center",
                       color="white" if data[mi, si] > data.max() * 0.6 else "black")
        
        ax.set_title(f"{title} 热力图", fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        path = os.path.join(out_dir, f"baseline_heatmap_{key}_{ts}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  [图表] {path}")
        plt.close(fig)


def print_results_table(all_results: dict):
    """打印结果表格"""
    header = f"{'模型':<16} {'场景':<12} {'通过车辆':<10} {'行程时间':<10} {'延误':<10} {'换道':<8} {'最大队列':<10} {'碰撞':<6}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for model_name in MODEL_ORDER:
        first = True
        for sc in SCENARIOS:
            if sc in all_results.get(model_name, {}):
                r = all_results[model_name][sc]
                prefix = model_name if first else ""
                print(f"{prefix:<16} {sc:<12} {r.get('total_vehicles', 0):<10} {r.get('avg_travel_time', 0):<10} "
                      f"{r.get('avg_delay', 0):<10} {r.get('lc_cnt', 0):<8} {r.get('max_queue', 0):<10} {r.get('collisions', 0):<6}")
                first = False
    print("=" * len(header))


# ====================== 主入口 ======================

def main():
    parser = argparse.ArgumentParser(description="从已保存的基线对比结果生成可视化图表")
    parser.add_argument("result_dir", type=str, nargs='?', default=None,
                        help="结果目录路径（包含 baseline_detail_*.csv 和 baseline_ts_*.pkl）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录（默认在结果目录下创建 charts_ 子目录）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用的结果目录")
    parser.add_argument("--types", type=str, default=None,
                        help="图表类型: bar,queue,speed,density,radar,heatmap,all（默认: all）")
    args = parser.parse_args()

    # 列出可用结果目录
    if args.list:
        dirs = find_result_dirs()
        if not dirs:
            print("未找到基线对比结果目录。")
            return
        print("可用的结果目录：")
        for d in dirs:
            # 统计目录中的 CSV/PKL 文件数
            csv_count = len(glob.glob(os.path.join(d, "baseline_detail_*.csv")) or glob.glob(os.path.join(d, "baseline_results_*.csv")))
            pkl_count = len(glob.glob(os.path.join(d, "baseline_ts_*.pkl")))
            status = "✓ 有数据" if csv_count > 0 else "○ 空目录"
            print(f"  {d}  [{status}] (CSV:{csv_count}, PKL:{pkl_count})")
        return

    if not args.result_dir:
        print("请指定结果目录。使用 --list 查看可用目录。")
        return

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print(f"[错误] 目录不存在: {result_dir}")
        return

    # 加载数据
    print(f"加载结果数据: {result_dir}")
    all_results = load_results_from_dir(result_dir)
    if all_results is None:
        return

    # 打印结果表格
    print_results_table(all_results)

    # 输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(result_dir, f"charts_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n图表输出目录: {out_dir}")

    # 选择图表类型
    chart_types = args.types.split(",") if args.types else ["all"]
    chart_any = "all" in chart_types

    if chart_any or "bar" in chart_types:
        print("\n生成分组柱状图...")
        plot_comparison_bar(all_results, out_dir, ts)

    if chart_any or "queue" in chart_types:
        print("生成队列长度时序图...")
        plot_queue_timeseries(all_results, out_dir, ts)

    if chart_any or "speed" in chart_types:
        print("生成速度时序图...")
        plot_speed_timeseries(all_results, out_dir, ts)

    if chart_any or "density" in chart_types:
        print("生成高密度场景对比图...")
        plot_high_density(all_results, out_dir, ts)

    if chart_any or "radar" in chart_types:
        print("生成雷达综合对比图...")
        plot_radar(all_results, out_dir, ts)

    if chart_any or "heatmap" in chart_types:
        print("生成热力图...")
        plot_heatmap(all_results, out_dir, ts)

    print(f"\n完成！所有图表已保存至: {out_dir}")


if __name__ == "__main__":
    main()
