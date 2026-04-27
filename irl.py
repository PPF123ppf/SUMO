"""
irl.py — 最大熵逆强化学习（MaxEnt IRL）
========================================
从 AD4CHE 自然驾驶数据中学出 PAYOFF_WEIGHTS。

流程：
  1. 读取 AD4CHE tracks.csv → 提取换道轨迹片段
  2. 对每个片段调用 compute_features() 计算专家特征期望
  3. 用当前 PAYOFF_WEIGHTS 在仿真中 rollout → 学习者特征期望
  4. 梯度上升更新权重，使两者一致

用法：
  python irl.py --data-dir data/AD4CHE --iterations 100
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ─── 确保能找到 game_lane_change ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_lane_change as glc


# ====================== 1. AD4CHE 数据加载 ======================

def load_ad4che_tracks(data_dir: str) -> pd.DataFrame:
    """加载 AD4CHE 某一段录制的 tracks.csv（支持递归查找）。"""
    candidates = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith("_tracks.csv"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 *_tracks.csv 文件")
    # 加载所有找到的录制并合并
    frames = []
    for path in sorted(candidates):
        df = pd.read_csv(path)
        df.rename(columns={"id": "trackId"}, inplace=True)
        rec_id = os.path.basename(os.path.dirname(path))
        df["recording"] = rec_id
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    print(f"  加载 {len(candidates)} 段录制，共 {len(result)} 帧，{result['trackId'].nunique()} 辆车")
    return result


def load_ad4che_meta(data_dir: str) -> pd.DataFrame:
    """加载所有 tracksMeta.csv。"""
    candidates = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith("_tracksMeta.csv"):
                candidates.append(os.path.join(root, f))
    frames = []
    for path in sorted(candidates):
        df = pd.read_csv(path)
        df["recording"] = os.path.basename(os.path.dirname(path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_lane_change_episodes(tracks: pd.DataFrame,
                                  tracks_meta: pd.DataFrame,
                                  window: int = 30) -> list:
    """
    从轨迹数据中提取换道片段（内存高效版：按 recording 分组处理）。

    参数:
        tracks: 完整轨迹表
        tracks_meta: 车辆元数据表
        window: 换道前后各取 frame 数（默认 30 = 1s @ 30fps）

    返回:
        episodes: list of dict
    """
    lc_vehicles = tracks_meta[tracks_meta["numLaneChanges"] > 0]
    lc_ids = set(lc_vehicles["id"].values)
    episodes = []

    # 按 recording 分组处理（避免大表过滤）
    for rec, group in tracks.groupby("recording"):
        for vid in lc_ids:
            veh = group[group["trackId"] == vid]
            if len(veh) < window:
                continue
            veh = veh.sort_values("frame")
            lane_changes = detect_lane_changes(veh)
            for lc_frame in lane_changes:
                seg = veh[
                    (veh["frame"] >= lc_frame - window) &
                    (veh["frame"] <= lc_frame + window)
                ].copy()
                if len(seg) < window * 0.5:
                    continue
                episodes.append({
                    "frames": seg,
                    "lc_frame": lc_frame,
                    "track_id": vid,
                })
    return episodes


def detect_lane_changes(veh_tracks: pd.DataFrame) -> list:
    """检测车辆换道时刻（laneId 变化的帧）。"""
    if "laneId" not in veh_tracks.columns:
        return []
    lanes = veh_tracks["laneId"].values
    changes = []
    for i in range(1, len(lanes)):
        if lanes[i] != lanes[i - 1]:
            changes.append(int(veh_tracks.iloc[i]["frame"]))
    return changes


# ====================== 2. 从数据中提取特征 ======================

def compute_expert_features(episodes: list) -> np.ndarray:
    """
    对每个换道片段，在换道前/后时刻计算特征向量。

    返回: (n_samples, n_features) 的专家特征矩阵
    """
    all_feats = []
    for ep in episodes:
        frames = ep["frames"]
        lc_frame = ep["lc_frame"]

        # 取换道前 0.3s（约 9 帧）和换道后 0.3s 的状态
        before = frames[frames["frame"] < lc_frame].tail(9)
        after = frames[frames["frame"] >= lc_frame].head(9)

        for group, label in [(before, 1), (after, 0)]:
            for _, row in group.iterrows():
                feats = row_to_features(row, label)
                all_feats.append(feats)

    return np.array(all_feats)


def row_to_features(row: pd.Series, action: int) -> np.ndarray:
    """
    将 AD4CHE 的一帧数据映射到 compute_features 同构的特征空间。

    action: 1 = 执行换道, 0 = 不换道

    返回: [eff, safe, coop, lc_cost]
    """
    # 纵向速度：取绝对值（AD4CHE 双向车道）
    v = max(abs(float(row.get("xVelocity", 0))), 1e-6)
    vmax = max(v * 1.3, 25.0)

    # 效率特征
    speed_ratio = float(np.clip(v / vmax, 0.0, 1.0))
    urgency = 0.0  # AD4CHE 无事故场景
    eff = 0.55 * speed_ratio + 0.45 * urgency

    # 安全特征：用 dhw（车头时距）或 ttc
    # AD4CHE 中 ttc 负数表示无碰撞风险
    dhw = float(row.get("dhw", 10.0))
    if dhw <= 0 or dhw > 100:
        dhw = 10.0
    safe = float(np.clip(dhw / 30.0, 0.0, 1.0))  # 30m 以上视为安全

    # 协同特征
    has_follower = int(pd.notna(row.get("followingId", None)))
    coop = 0.18 if has_follower else 0.0

    lc_cost = 1.0 if action == 1 else 0.0
    return np.array([eff, safe, coop, lc_cost])


# ====================== 3. 仿真 rollout ======================

def learner_rollout(weights: np.ndarray,
                    n_cav: int = 120,
                    sim_steps: int = 3600,
                    n_rollouts: int = 5) -> np.ndarray:
    """
    用当前权重运行仿真，通过 _irl_feature_log 收集特征。

    返回: (n_samples, n_features) 特征矩阵
    """
    import copy
    glc.PAYOFF_WEIGHTS["informed"] = weights.copy()
    os.environ['SIM_STEPS'] = str(sim_steps)

    all_feats = []
    for _ in range(n_rollouts):
        glc.run_once(n_cav, "IRL_rollout")
        # 从特征日志快照中提取每个决策点的特征
        for payoff, feats in glc._irl_feature_log_snapshot:
            chosen_action = 0 if payoff[0].max() > payoff[1].max() else 1
            best_fa = np.argmax(payoff[chosen_action])
            fv = feats[(chosen_action, best_fa)]
            all_feats.append(fv)
    return np.array(all_feats) if all_feats else np.zeros((1, 4))


# ====================== 4. 最大熵 IRL 核心 ======================

def maxent_irl(expert_features: np.ndarray,
               initial_weights: np.ndarray,
               learning_rate: float = 0.01,
               n_iterations: int = 100,
               l2_reg: float = 0.001) -> tuple:
    """
    最大熵逆强化学习主循环。

    参数:
        expert_features: (N, D) 专家特征矩阵
        initial_weights: (D,) 初始权重
        learning_rate: 梯度上升步长
        n_iterations: 迭代次数

    返回:
        learned_weights: 学习后的权重
        loss_history: 每步 loss
    """
    weights = initial_weights.copy()
    expert_mean = expert_features.mean(axis=0)
    loss_history = []

    print(f"  专家特征期望: {expert_mean}")
    print(f"  初始权重: {weights}")

    for it in range(n_iterations):
        # 用当前权重采集学习者特征
        learner_feats = learner_rollout(weights, n_rollouts=2)
        learner_mean = learner_feats.mean(axis=0)

        # 梯度 = 专家期望 - 学习者期望（最大熵 IRL 核心更新式）
        grad = expert_mean - learner_mean - l2_reg * weights

        # 梯度上升
        weights += learning_rate * grad
        weights = np.clip(weights, 0.0, 2.0)

        # Loss = 负对数似然（监控用）
        loss = -np.dot(expert_mean, weights) + np.log(
            np.sum(np.exp(np.dot(learner_feats, weights)))
        )
        loss_history.append(loss)

        if it % 10 == 0 or it == n_iterations - 1:
            print(f"  iter {it:3d}: loss={loss:.4f}  weights={[f'{w:.3f}' for w in weights]}")

    return weights, loss_history


# ====================== 5. 应用到仿真 ======================

def apply_weights(weights_sudden: np.ndarray,
                  weights_informed: np.ndarray):
    """将学习到的权重写回 game_lane_change 模块。"""
    glc.PAYOFF_WEIGHTS["sudden"] = weights_sudden.copy()
    glc.PAYOFF_WEIGHTS["informed"] = weights_informed.copy()
    print(f"\n权重已更新:")
    print(f"  sudden:   {weights_sudden}")
    print(f"  informed: {weights_informed}")
    print(f"  (特征顺序: {glc.FEATURE_NAMES})")


# ====================== 6. 完整流程 ======================

def save_weights(path: str = "irl_weights.npz"):
    """保存当前权重到文件。"""
    np.savez(path,
             sudden=glc.PAYOFF_WEIGHTS["sudden"],
             informed=glc.PAYOFF_WEIGHTS["informed"])
    print(f"  权重已保存: {path}")


def load_weights(path: str = "irl_weights.npz") -> bool:
    """从文件加载权重。"""
    try:
        data = np.load(path)
        glc.PAYOFF_WEIGHTS["sudden"] = data["sudden"]
        glc.PAYOFF_WEIGHTS["informed"] = data["informed"]
        print(f"  权重已加载: {path}")
        print(f"    sudden={data['sudden']}")
        print(f"    informed={data['informed']}")
        return True
    except (FileNotFoundError, OSError):
        return False


def run_irl(data_dir: str, iterations: int = 100, resume: bool = False):
    """完整的 IRL 流程。"""
    print("=" * 60)
    print("  最大熵逆强化学习 (MaxEnt IRL)")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据: {data_dir}")
    tracks = load_ad4che_tracks(data_dir)
    meta = load_ad4che_meta(data_dir)
    print(f"  车辆: {len(meta)}, 帧: {len(tracks)}")

    # 2. 提取换道片段
    print(f"\n[2/4] 提取换道片段...")
    episodes = extract_lane_change_episodes(tracks, meta)
    print(f"  找到 {len(episodes)} 个换道片段")

    if not episodes:
        print("  [错误] 没有找到换道片段，请检查数据路径")
        return

    # 3. 计算专家特征期望
    print(f"\n[3/4] 计算专家特征期望...")
    expert_feats = compute_expert_features(episodes)
    print(f"  专家特征矩阵: {expert_feats.shape}")

    # 4. 运行 IRL
    print(f"\n[4/4] 运行 IRL ({iterations} 次迭代)...")
    initial = glc.PAYOFF_WEIGHTS["informed"].copy()
    learned_weights, loss = maxent_irl(
        expert_features=expert_feats,
        initial_weights=initial,
        n_iterations=args.iterations,
    )

    # 5. 应用
    apply_weights(
        weights_sudden=glc.PAYOFF_WEIGHTS["sudden"],
        weights_informed=learned_weights,
    )
    print("\n完成。新权重已写入 PAYOFF_WEIGHTS")
    save_weights()


# ====================== 命令行入口 ======================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="最大熵逆强化学习 (MaxEnt IRL)")
    parser.add_argument("--data-dir", type=str,
                        default="data/AD4CHE_dataset_V1.0/AD4CHE_dataset_V1.0/AD4CHE_Data_V1.0",
                        help="AD4CHE 数据根目录（自动遍历所有子目录中的 *_tracks.csv）")
    parser.add_argument("--iterations", type=int, default=100,
                        help="IRL 迭代次数")
    parser.add_argument("--lr", type=float, default=0.03,
                        help="学习率")
    parser.add_argument("--resume", type=str, default=None,
                        help="从权重文件恢复（如 irl_weights.npz）")
    args = parser.parse_args()

    if args.resume:
        load_weights(args.resume)
        print(f"恢复权重后继续训练 {args.iterations} 轮")

    run_irl(data_dir=args.data_dir, iterations=args.iterations)
