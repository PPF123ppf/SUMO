"""从 parallel_final 结果生成对比图表"""
import os, glob, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_pen(s):
    for p in ['100%CAV','70%CAV','50%CAV','30%CAV','10%CAV','0%CAV']:
        if p in str(s): return p
    return ''
def get_density(s):
    for d in ['1200','2000','2800','3600']:
        if d in str(s): return int(d)
    return 0

def load_all_results(base='results/parallel_v5'):
    models = ['Game (Ours)','SUMO Default','Rule-Based','No-V2X']
    all_rows = []
    for mname in models:
        d = os.path.join(base, mname)
        csvs = sorted(glob.glob(os.path.join(d, 'baseline_results_*.csv')))
        if not csvs: continue
        df = pd.read_csv(csvs[-1])
        df['model_name'] = mname
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

df = load_all_results()
df['pen'] = df['scenario'].apply(get_pen)
df['density'] = df['scenario'].apply(get_density)
outdir = 'results/figures'
os.makedirs(outdir, exist_ok=True)

models = ['Game (Ours)', 'SUMO Default', 'Rule-Based', 'No-V2X']
pens_order = ['0%CAV','10%CAV','30%CAV','50%CAV','70%CAV','100%CAV']
densities = [1200,2000,2800,3600]
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

# === 图1: 渗透率扫描热图 ===
game = df[df['model_name']=='Game (Ours)']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, metric in enumerate(['total_vehicles', 'max_queue']):
    data = np.zeros((len(pens_order), len(densities)))
    for i, pen in enumerate(pens_order):
        for j, d in enumerate(densities):
            r = game[(game['pen']==pen)&(game['density']==d)]
            if len(r): data[i,j] = float(r[metric].values[0])
    cmap = 'YlOrRd' if metric=='max_queue' else 'YlGn'
    im = axes[idx].imshow(data, cmap=cmap, aspect='auto')
    axes[idx].set_xticks(range(len(densities)))
    axes[idx].set_xticklabels([f'{d}' for d in densities])
    axes[idx].set_yticks(range(len(pens_order)))
    axes[idx].set_yticklabels([p.replace('%CAV','') for p in pens_order])
    axes[idx].set_title('通过车辆数' if metric=='total_vehicles' else '最大队列')
    axes[idx].set_xlabel('密度(pcu/h)')
    axes[idx].set_ylabel('CAV渗透率(%)')
    for i2 in range(data.shape[0]):
        for j2 in range(data.shape[1]):
            axes[idx].text(j2, i2, f'{data[i2,j2]:.0f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=axes[idx])
plt.suptitle('Game (Ours) 渗透率扫描', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'penetration_sweep.png'), dpi=200, bbox_inches='tight')
plt.close()
print('1/4: penetration_sweep.png')

# === 图2: 全模型对比 ===
all100 = df[df['pen']=='100%CAV']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
metrics = [('total_vehicles', '通过车辆数'), ('avg_travel_time', '平均行程时间(s)'),
           ('lc_cnt', '换道次数'), ('max_queue', '最大队列长度')]
for idx, (metric, title) in enumerate(metrics):
    ax = axes[idx//2][idx%2]
    x = np.arange(len(densities)); w = 0.2
    for mi, model in enumerate(models):
        vals = []
        for d in densities:
            r = all100[(all100['model_name']==model)&(all100['density']==d)]
            vals.append(float(r[metric].values[0]) if len(r) else 0)
        ax.bar(x + mi*w, vals, w, label=model, color=colors[mi])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([f'{d}pcu' for d in densities])
    ax.set_title(title, fontsize=12); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.suptitle('全模型对比 @ 100% CAV', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print('2/4: model_comparison.png')

# === 图3: 渗透率折线图 ===
fig, ax = plt.subplots(figsize=(12, 6))
for d in densities:
    r = game[game['density']==d].sort_values('pen')
    if len(r):
        ax.plot(range(len(r)), r['total_vehicles'].tolist(), 'o-', label=f'{d}pcu/h', linewidth=2)
ax.set_xticks(range(len(pens_order)))
ax.set_xticklabels([p.replace('%CAV','') for p in pens_order])
ax.set_xlabel('CAV 渗透率 (%)'); ax.set_ylabel('通过车辆数')
ax.set_title('Game (Ours) 渗透率 vs 通过量'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'penetration_throughput.png'), dpi=200, bbox_inches='tight')
plt.close()
print('3/4: penetration_throughput.png')

# === 图4: 换道活跃度渗透率折线图 ===
fig, ax = plt.subplots(figsize=(12, 6))
for d in densities:
    r = game[game['density']==d].sort_values('pen')
    if len(r):
        vals = r['lc_cnt'].tolist()
        ax.plot(range(len(vals)), vals, 'o-', label=f'{d}pcu/h', linewidth=2)
ax.set_xticks(range(len(pens_order)))
ax.set_xticklabels([p.replace('%CAV','') for p in pens_order])
ax.set_xlabel('CAV 渗透率 (%)'); ax.set_ylabel('换道次数')
ax.set_title('Game (Ours) 渗透率 vs 换道活跃度'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'penetration_lc.png'), dpi=200, bbox_inches='tight')
plt.close()
print('4/4: penetration_lc.png')

print(f'\n图片已保存至 {outdir}/')
