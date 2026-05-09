"""
train_drl.py — PPO 深度强化学习训练脚本
=========================================
基于 stable-baselines3，在 LaneChangeEnv 中训练 PPO 换道决策策略。

用法:
  python train_drl.py                          # 默认训练
  python train_drl.py --total-timesteps 50000  # 自定义步数
  python train_drl.py --evaluate               # 评估已训练模型

输出:
  models/ppo_lanechange.zip    — 训练好的策略
  tensorboard_logs/            — 训练曲线 (用 tensorboard --logdir tensorboard_logs 查看)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sumo_env import LaneChangeEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


MODEL_DIR = "models"
LOG_DIR   = "training_logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(seed: int = 0):
    """环境工厂函数 (用于 DummyVecEnv)."""
    def _init():
        env = LaneChangeEnv()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(total_timesteps: int = 100000,
          learning_rate: float = 3e-4,
          n_steps: int = 1024,
          batch_size: int = 64,
          verbose: int = 1):
    """
    PPO 训练主函数.

    参数:
        total_timesteps: 总训练步数 (default 100k)
        learning_rate:   PPO 学习率
        n_steps:         每轮更新前的步数
        batch_size:      小批量大小
    """
    print("=" * 60)
    print("  PPO Lane-Change Decision Training")
    print("=" * 60)
    print(f"  环境:     LaneChangeEnv (8-dim state, 2-action discrete)")
    print(f"  算法:     PPO (clip-ratio, GAE-Lambda)")
    print(f"  总计步数: {total_timesteps}")
    print()

    env = DummyVecEnv([make_env(seed=42)])
    # eval_env 需要 reset/step 接口与 train 一致
    eval_env = DummyVecEnv([make_env(seed=999)])

    # ── 回调 ──
    reward_threshold = 0.20  # 平均奖励超过 0.2 即认为收敛
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=500,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    # ── PPO 模型 ──
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
    )

    # ── 训练 ──
    print("  开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    # ── 保存 ──
    model_path = os.path.join(MODEL_DIR, "ppo_lanechange")
    model.save(model_path)
    print(f"\n  模型已保存: {model_path}.zip")

    # ── 最终评估 ──
    evaluate(model_path)


def evaluate(model_path: str = os.path.join(MODEL_DIR, "ppo_lanechange"),
             n_episodes: int = 50):
    """加载训练好的模型并评估."""
    if not os.path.exists(f"{model_path}.zip") and not os.path.exists(model_path):
        print(f"  [错误] 模型不存在: {model_path}")
        return

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"  [错误] 加载模型失败: {e}")
        return

    env = LaneChangeEnv()
    total_rewards = []
    stay_rates = []
    change_rates = []

    print(f"\n  评估模型 ({n_episodes} episodes)...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        actions = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action.item() if hasattr(action, 'item') else action)
            actions.append(action_int)
            obs, reward, terminated, truncated, _ = env.step(action_int)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        if actions:
            stay_rates.append(sum(1 for a in actions if a == 0) / len(actions))
            change_rates.append(sum(1 for a in actions if a == 1) / len(actions))

    print(f"  {'='*40}")
    print(f"  平均奖励:        {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"  保持车道率:      {np.mean(stay_rates):.2%}")
    print(f"  换道率:          {np.mean(change_rates):.2%}")
    print(f"  成功换道率:      由 SUMO 仿真评估")
    print(f"  {'='*40}")


def main():
    parser = argparse.ArgumentParser(description="PPO 换道决策训练")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="总训练步数 (default: 100000)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO 学习率 (default: 3e-4)")
    parser.add_argument("--evaluate", action="store_true",
                        help="仅评估已有模型，不训练")
    parser.add_argument("--model-path", type=str, default=None,
                        help="评估时指定模型路径")
    args = parser.parse_args()

    if args.evaluate:
        model_path = args.model_path or os.path.join(MODEL_DIR, "ppo_lanechange")
        evaluate(model_path)
    else:
        train(
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
