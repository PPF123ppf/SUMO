"""
快速 IRL 训练（子集数据，30 轮，每轮 2 rollout × 1200 步）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import game_lane_change as glc
import irl

# 子集数据加载
data_dir = 'data/AD4CHE_dataset_V1.0/AD4CHE_dataset_V1.0/AD4CHE_Data_V1.0'
sub_dirs = sorted(os.listdir(data_dir))[:5]
print(f"使用 {len(sub_dirs)} 段录制")

frames = []
for d in sub_dirs:
    for f in os.listdir(os.path.join(data_dir, d)):
        if f.endswith('_tracks.csv'):
            path = os.path.join(data_dir, d, f)
            df = pd.read_csv(path)
            df.rename(columns={'id': 'trackId'}, inplace=True)
            df['recording'] = d
            frames.append(df)
tracks = pd.concat(frames, ignore_index=True)
print(f"  加载 {len(frames)} 段录制，共 {len(tracks)} 帧，{tracks['trackId'].nunique()} 辆车")

meta_frames = []
for d in sub_dirs:
    for f in os.listdir(os.path.join(data_dir, d)):
        if f.endswith('_tracksMeta.csv'):
            path = os.path.join(data_dir, d, f)
            df = pd.read_csv(path)
            df['recording'] = d
            meta_frames.append(df)
meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()

episodes = irl.extract_lane_change_episodes(tracks, meta)
print(f"  换道片段: {len(episodes)}")

expert_feats = irl.compute_expert_features(episodes)
print(f"  专家特征: {expert_feats.shape} (8维)")

expert_mean = expert_feats.mean(axis=0)
print(f"  专家特征期望: {[f'{v:.3f}' for v in expert_mean]}")

# IRL 训练
print(f"\n开始 IRL 训练（30 轮）...")
glc.CAV_PENETRATION = 1.0  # 全 CAV 环境训练
os.environ['SIM_STEPS'] = '1200'  # 120s 每轮

weights = glc.PAYOFF_WEIGHTS["informed"].copy()
print(f"  初始权重: {[f'{w:.3f}' for w in weights]}")

# 简化版 IRL 循环（直接内联）
lr = 0.02
l2_reg = 0.001
loss_history = []

for it in range(30):
    learner_feats = irl.learner_rollout(weights, n_cav=120, sim_steps=1200, n_rollouts=2)
    learner_mean = learner_feats.mean(axis=0)

    grad = expert_mean - learner_mean - l2_reg * weights
    weights += lr * grad
    weights = np.clip(weights, 0.0, 2.0)

    loss = -np.dot(expert_mean, weights) + np.log(
        np.sum(np.exp(np.dot(learner_feats, weights))) + 1e-10
    )
    loss_history.append(loss)

    if it % 5 == 0 or it == 29:
        print(f"  iter {it:2d}: loss={loss:.4f}  weights={[f'{w:.3f}' for w in weights]}")

# 应用学到的权重
glc.PAYOFF_WEIGHTS["informed"] = weights.copy()
glc.PAYOFF_WEIGHTS["sudden"] = weights.copy() * 0.8  # 突发期更保守

print(f"\n最终权重: {[f'{w:.4f}' for w in weights]}")
print(f"  特征: {glc.FEATURE_NAMES}")
print(f"  Loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")

# 保存
irl.save_weights("irl_weights_v2.npz")
print("权重已保存: irl_weights_v2.npz")
