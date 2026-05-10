"""
plot_publication.py — 出版级可视化图表生成
===========================================
生成三组图表：
  1. 对比柱状图 (PDF)
  2. 3D 曲面图 (CAV渗透率 × 密度 → 指标)
  3. 事故区车辆轨迹热力图

用法:
  python plot_publication.py                         # 全部生成
  python plot_publication.py --charts 1 2             # 仅图1+2
  python plot_publication.py --no-show                # 不显示窗口
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── LaTeX 字体配置 ───
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['pdf.fonttype'] = 42  # 可编辑文本
rcParams['ps.fonttype'] = 42

# ─── 统一配色 ───
COLORS = {
    'Game (Ours)': '#2166AC',
    'DRL (PPO)':   '#D6604D',
    'No-V2X':      '#4DAF4A',
    'SUMO Default':'#FF7F00',
    'Rule-Based':  '#984EA3',
}
COLOR_LIST = ['#2166AC', '#D6604D', '#4DAF4A', '#FF7F00', '#984EA3']
COLOR_CAV = plt.cm.viridis(np.linspace(0.2, 0.9, 6))

OUT_DIR = 'results/figures/publication'
os.makedirs(OUT_DIR, exist_ok=True)

ACCIDENT_TIME = 90.0
ACCIDENT_START = 3000.0
ACCIDENT_END = 3200.0


# ═══════════════════════════════════════════
# 图 1: 出版级对比柱状图
# ═══════════════════════════════════════════

def fig1_comparison(csv_path: str):
    """五模型 × 六渗透率 @ 3600pcu/h 对比."""
    df = pd.read_csv(csv_path)
    mask = df['scenario'].str.contains('3600')
    d = df[mask].copy()
    cav_rates = sorted(d['cav_rate'].unique())
    models = ['Game (Ours)', 'DRL (PPO)', 'No-V2X', 'SUMO Default', 'Rule-Based']

    metrics = [
        ('total_vehicles', 'Number of Vehicles\nCompleted', 'vehicles'),
        ('avg_delay', 'Average Delay (s)', 's'),
        ('max_queue', 'Max Queue Length\n(vehicles)', 'vehicles'),
        ('lc_cnt', 'Number of Lane\nChanges', 'count'),
        ('collisions', 'Collisions', 'count'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    axes_flat = axes.flatten()
    for idx, ax in enumerate(axes_flat[:5]):
        metric, title, unit = metrics[idx]
        x = np.arange(len(cav_rates))
        n = len(models)
        width = 0.70 / n

        for mi, model in enumerate(models):
            vals = []
            for cav in cav_rates:
                r = d[(d['cav_rate']==cav) & (d['model']==model)]
                vals.append(r[metric].values[0] if len(r) else 0)
            offset = (mi - (n-1)/2) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                         color=COLORS[model], label=model if idx == 0 else "",
                         edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{r}%' for r in cav_rates], fontsize=11)
        ax.set_xlabel('CAV Penetration Rate', fontsize=12)
        ax.set_ylabel(title, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)

    axes_flat[5].axis('off')
    fig.legend(models, loc='lower center', ncol=5, fontsize=11,
               frameon=True, edgecolor='#CCCCCC')
    fig.suptitle('Performance Comparison Across CAV Penetration Rates\n'
                 '(3600 pcu/h, 3-run average)',
                 fontsize=15, fontweight='bold', y=1.02)

    path = os.path.join(OUT_DIR, 'fig1_comparison.pdf')
    fig.savefig(path, dpi=300)
    print(f'  Saved: {path}')
    plt.close(fig)


# ═══════════════════════════════════════════
# 图 2: 3D 曲面图
# ═══════════════════════════════════════════

def fig2_surface(csv_path: str):
    """CAV 渗透率 × 交通密度 → 延误 / 排队 / 换道 3D 曲面."""
    df = pd.read_csv(csv_path)
    cav_rates = sorted(df['cav_rate'].unique())
    # 从 scenario 列提取密度档 (1200/2000/2800/3600)
    density_map = {'1200pcu/h': 1200, '2000pcu/h': 2000,
                   '2800pcu/h': 2800, '3600pcu/h': 3600}

    models_3d = ['Game (Ours)', 'DRL (PPO)', 'No-V2X']
    metrics_3d = [
        ('avg_delay', 'Average Delay (s)'),
        ('max_queue', 'Max Queue Length (veh)'),
        ('lc_cnt', 'Lane Changes'),
    ]

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    plot_idx = 1
    for metric, title in metrics_3d:
        for mi, model in enumerate(models_3d):
            ax = fig.add_subplot(3, 3, plot_idx, projection='3d')
            X, Y = np.meshgrid(cav_rates, list(density_map.values()))
            Z = np.zeros_like(X, dtype=float)

            for i, cav in enumerate(cav_rates):
                for j, (slbl, den) in enumerate(density_map.items()):
                    r = df[(df['cav_rate']==cav) & (df['model']==model) & (df['scenario']==slbl)]
                    Z[j, i] = r[metric].values[0] if len(r) else 0

            surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                                   edgecolor='none', alpha=0.9,
                                   antialiased=True)
            ax.set_xlabel('CAV Rate (%)', fontsize=10, labelpad=8)
            ax.set_ylabel('Density (pcu/h)', fontsize=10, labelpad=8)
            ax.set_zlabel(title.split('(')[0].strip(), fontsize=10, labelpad=8)
            ax.tick_params(labelsize=9)
            ax.view_init(elev=25, azim=-60)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

            if plot_idx <= 3:
                ax.set_title(f'{model}', fontsize=12, fontweight='bold', pad=15)
            if plot_idx % 3 == 2:
                cb = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
                cb.set_label(title, fontsize=9)

            plot_idx += 1

    fig.suptitle('3D Surface: CAV Rate × Traffic Density → Performance Metrics\n'
                 '(3-run averaged)',
                 fontsize=15, fontweight='bold', y=1.02)

    path = os.path.join(OUT_DIR, 'fig2_surface.pdf')
    fig.savefig(path, dpi=300)
    print(f'  Saved: {path}')
    plt.close(fig)


# ═══════════════════════════════════════════
# 图 3: 轨迹热力图 (时空图)
# ═══════════════════════════════════════════

def fig3_trajectory(fcd_path: str = None):
    """事故区车辆时空轨迹热力图.
    从 FCD XML 解析车辆位置, 绘制 time-space 图谱.
    """
    if fcd_path and not os.path.exists(fcd_path):
        # 尝试查找已有 FCD
        candidates = [f for f in os.listdir('.') if f.startswith('fcd_') and f.endswith('.xml')]
        if candidates:
            fcd_path = candidates[0]
            print(f'  Using existing FCD: {fcd_path}')
        else:
            print('  No FCD data found, generating trajectory from tripinfo...')
            fcd_path = None

    if fcd_path:
        _plot_trajectory_from_fcd(fcd_path)
    else:
        _plot_trajectory_fallback()


def _plot_trajectory_from_fcd(fcd_path: str):
    """从 FCD XML 绘制时空图."""
    import xml.etree.ElementTree as ET
    print(f'  Parsing FCD: {fcd_path}')
    tree = ET.parse(fcd_path)
    root = tree.getroot()

    # 收集车辆轨迹
    vehicle_data = {}
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time', 0))
        if time < 80 or time > 200:  # 聚焦事故时段
            continue
        for vehicle in timestep.findall('vehicle'):
            vid = vehicle.get('id')
            x = float(vehicle.get('x', 0))
            speed = float(vehicle.get('speed', 0))

            if vid not in vehicle_data:
                vehicle_data[vid] = {'t': [], 'x': [], 'v': []}
            vehicle_data[vid]['t'].append(time)
            vehicle_data[vid]['x'].append(x)
            vehicle_data[vid]['v'].append(speed)

    # 过滤掉数据点太少的车
    vehicle_data = {k: v for k, v in vehicle_data.items() if len(v['t']) > 20}

    print(f'  Loaded {len(vehicle_data)} vehicles')

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # 绘制每条轨迹, 按速度着色
    for vid, data in vehicle_data.items():
        t = data['t']
        x = data['x']
        v = np.array(data['v'])
        # 将速度映射到颜色
        v_norm = np.clip(v / 30.0, 0, 1)  # max speed ~ 30 m/s
        # 分段绘制: 每段颜色不同
        for i in range(len(t) - 1):
            color = plt.cm.plasma(v_norm[i])
            ax.plot(t[i:i+2], x[i:i+2], color=color, linewidth=0.3, alpha=0.7)

    # 标注事故区
    ax.axhspan(ACCIDENT_START, ACCIDENT_END, alpha=0.15, color='red',
               label='Accident Zone (3000-3200m)')
    ax.axvline(x=ACCIDENT_TIME, color='red', linestyle='--', alpha=0.7,
               label=f'Accident at t={ACCIDENT_TIME}s')
    ax.axvline(x=ACCIDENT_TIME + 10, color='orange', linestyle=':', alpha=0.7,
               label='Global Broadcast Activated')

    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Position on Road (m)', fontsize=13)
    ax.set_title('Vehicle Trajectories Around Accident Zone\n'
                 '(Game Model, 70% CAV, 2800 pcu/h)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(linestyle='--', alpha=0.2)
    ax.set_xlim(80, 200)
    ax.set_ylim(2500, 3500)

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(0, 30))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Speed (m/s)', fontsize=12)

    path = os.path.join(OUT_DIR, 'fig3_trajectory.pdf')
    fig.savefig(path, dpi=300)
    print(f'  Saved: {path}')
    plt.close(fig)


def _plot_trajectory_fallback():
    """无 FCD 数据时, 从 CSV 数据生成指标汇总图."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Trajectory data not available.\nRun simulation with FCD output enabled.',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Vehicle Trajectory Heatmap', fontsize=14)
    path = os.path.join(OUT_DIR, 'fig3_trajectory.pdf')
    fig.savefig(path, dpi=300)
    print(f'  Saved (placeholder): {path}')
    plt.close(fig)


# ═══════════════════════════════════════════
# 辅助: 游戏模型渗透率曲线
# ═══════════════════════════════════════════

def fig4_game_penetration(csv_path: str):
    """Game 模型渗透率-关键指标折线图 (用于论文)."""
    df = pd.read_csv(csv_path)
    d = df[df['model'] == 'Game (Ours)'].copy()
    density_vals = {'1200pcu/h': '1200', '2000pcu/h': '2000',
                    '2800pcu/h': '2800', '3600pcu/h': '3600'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    metrics_plot = [
        ('total_vehicles', 'Vehicles Completed', 'count'),
        ('avg_delay', 'Average Delay', 's'),
        ('max_queue', 'Max Queue Length', 'vehicles'),
        ('lc_cnt', 'Lane Changes', 'count'),
    ]

    for idx, (metric, title, unit) in enumerate(metrics_plot):
        ax = axes[idx // 2][idx % 2]
        for den_label, den_short in density_vals.items():
            r = d[d['scenario'] == den_label].sort_values('cav_rate')
            if len(r):
                ax.plot(r['cav_rate'], r[metric], 'o-', linewidth=2,
                       markersize=6, label=f'{den_short} pcu/h')

        ax.set_xlabel('CAV Penetration Rate (%)', fontsize=12)
        ax.set_ylabel(f'{title} ({unit})', fontsize=12)
        ax.set_title(f'{title}', fontsize=13)
        ax.legend(fontsize=10, title='Density')
        ax.grid(linestyle='--', alpha=0.3)
        ax.set_xticks([0, 10, 30, 50, 70, 100])
        ax.set_xticklabels(['0%', '10%', '30%', '50%', '70%', '100%'])

    fig.suptitle('Game Model Performance Across CAV Penetration Rates\n'
                 '(3-run average)',
                 fontsize=15, fontweight='bold', y=1.02)

    path = os.path.join(OUT_DIR, 'fig4_game_penetration.pdf')
    fig.savefig(path, dpi=300)
    print(f'  Saved: {path}')
    plt.close(fig)


# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='出版级图表生成')
    parser.add_argument('--csv', type=str, default=None,
                        help='平均 CSV 路径 (自动查找最新)')
    parser.add_argument('--fcd', type=str, default=None,
                        help='FCD XML 路径')
    parser.add_argument('--charts', type=int, nargs='*', default=[1,2,3,4],
                        help='要生成的图表编号 (default: all)')
    args = parser.parse_args()

    # 自动查找最新平均 CSV
    csv_path = args.csv
    if csv_path is None:
        results_dir = 'results'
        cav_dirs = [d for d in os.listdir(results_dir)
                    if d.startswith('cav_scan_') and
                    os.path.exists(os.path.join(results_dir, d, 'cav_scan_avg_*.csv'))]
        # 直接找最新的
        import glob
        avg_files = sorted(glob.glob(os.path.join(results_dir, '**/cav_scan_avg_*.csv')),
                          reverse=True)
        if avg_files:
            csv_path = avg_files[0]
            print(f'Using: {csv_path}')
        else:
            print('No average CSV found!')
            return

    charts = set(args.charts)

    if 1 in charts:
        print('\n[1/4] Publication comparison chart...')
        fig1_comparison(csv_path)

    if 2 in charts:
        print('\n[2/4] 3D surface plots...')
        fig2_surface(csv_path)

    if 3 in charts:
        print('\n[3/4] Trajectory heatmap...')
        fig3_trajectory(args.fcd)

    if 4 in charts:
        print('\n[4/4] Game penetration curves...')
        fig4_game_penetration(csv_path)

    print(f'\nAll figures saved to: {OUT_DIR}/')
    print(f'  Files: {os.listdir(OUT_DIR)}')


if __name__ == '__main__':
    main()
