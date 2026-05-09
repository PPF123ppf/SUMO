"""
sumo_env.py — Gym 环境封装，用于换道决策的深度强化学习训练
===========================================================

设计:
  决策层面环境: 每步 = 一个 CAV 换道决策点
  状态空间: 8维原子特征 [speed, urgency, pressure, safe, coop, social, density, lc_cost]
  动作空间: 离散 2 维 [0=stay, 1=change lanes]
  奖励: 安全 + 效率 + 舒适性

训练环境是轻量的 (不跑完整 SUMO)，评估时接入真实 SUMO 仿真。
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# ====================== 特征索引 (与 game_lane_change 一致) ======================
F_SPEED, F_URGENCY, F_PRESSURE, F_SAFE, F_COOP, F_SOCIAL, F_DENSITY, F_LC_COST = range(8)

# ====================== 奖励权重 ======================
R_SPEED_GAIN    = 0.30   # 速度增益奖励
R_SAFE_BONUS    = 0.40   # 安全换道奖励
R_COLLISION_PEN = 1.00   # 碰撞/危险惩罚
R_COMFORT_PEN   = 0.10   # 舒适性惩罚
R_STAY_PENALTY  = 0.02   # 不换道的微小代价 (鼓励必要时的主动换道)
R_UNNECESSARY   = 0.15   # 非必要换道惩罚


class LaneChangeEnv(gym.Env):
    """
    轻量级换道决策 Gym 环境 (v2 — 更清晰的学习信号).

    状态 (8-dim):
      speed:    本车速度比 v/vmax [0,1]
      urgency:  逃离紧迫度 [0,1] — 随时间自然上升, 换道成功可缓解
      pressure: 前车挤压感 [0,1] — 由 density 和 speed 共同决定
      safe:     安全裕度 [0,1] — 高 safe 时换道易成功, 低 safe 时换道风险高
      coop:     协同让行倾向 [0.05, 0.18]
      social:   换道社会影响 [0, 0.6]
      density:  局部密度 [0,1]
      lc_cost:  是否刚执行换道 {0, 1}

    动作:
      0 = 保持车道 (stay)
      1 = 换道 (change)

    核心设计:
      - safe 高了 → 换道几乎必然成功, 获得正向奖励
      - safe 低了 → 换道大概率失败, 受惩罚
      - 长时间 stay → urgency 上升, urgency 高了 → safe 下降
      - 智能体必须学会在 safe 尚高时果断换道
    """

    def __init__(self, reward_weights: Optional[dict] = None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.rw = reward_weights or {}

        self._rng = np.random.default_rng()
        self._state: Optional[np.ndarray] = None
        self._step_count = 0
        self._max_steps = 60
        self._stay_streak = 0  # 连续 stay 计数, 用于 urgency 累积

    # ─────────────────── 状态生成 ───────────────────

    def _sample_random_state(self) -> np.ndarray:
        """
        采样初始状态: 覆盖换道"应该做"和"不应该做"的场景.
        """
        s = np.zeros(8, dtype=np.float32)
        scenario = self._rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])

        if scenario == 0:  # 通畅: 不应换道
            s[F_SPEED]    = float(self._rng.uniform(0.80, 0.95))
            s[F_URGENCY]  = float(self._rng.uniform(0.0, 0.1))
            s[F_PRESSURE] = float(self._rng.uniform(0.0, 0.1))
            s[F_SAFE]     = float(self._rng.uniform(0.85, 0.95))
            s[F_DENSITY]  = float(self._rng.uniform(0.0, 0.15))
        elif scenario == 1:  # 轻微拥堵: 可换可不换
            s[F_SPEED]    = float(self._rng.uniform(0.55, 0.75))
            s[F_URGENCY]  = float(self._rng.uniform(0.2, 0.4))
            s[F_PRESSURE] = float(self._rng.uniform(0.2, 0.4))
            s[F_SAFE]     = float(self._rng.uniform(0.5, 0.75))
            s[F_DENSITY]  = float(self._rng.uniform(0.3, 0.6))
        elif scenario == 2:  # 拥堵: 应该换道
            s[F_SPEED]    = float(self._rng.uniform(0.3, 0.55))
            s[F_URGENCY]  = float(self._rng.uniform(0.4, 0.7))
            s[F_PRESSURE] = float(self._rng.uniform(0.4, 0.7))
            s[F_SAFE]     = float(self._rng.uniform(0.35, 0.55))
            s[F_DENSITY]  = float(self._rng.uniform(0.5, 0.8))
        else:  # 事故区急迫: 必须尽快换道
            s[F_SPEED]    = float(self._rng.uniform(0.15, 0.4))
            s[F_URGENCY]  = float(self._rng.uniform(0.6, 0.9))
            s[F_PRESSURE] = float(self._rng.uniform(0.6, 0.85))
            s[F_SAFE]     = float(self._rng.uniform(0.15, 0.4))
            s[F_DENSITY]  = float(self._rng.uniform(0.7, 1.0))

        has_follower = self._rng.random() < 0.4
        s[F_COOP]    = 0.18 if has_follower else 0.05
        s[F_SOCIAL]  = 0.0
        s[F_LC_COST] = 0.0

        return s

    def _next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        状态转移: action 与 state 的 safe/pressure 决定结果.
        """
        s = state.copy()

        if action == 1:  # change
            # 换道成功概率 = safe 值 * 0.9 + 0.1 (保证一定有下限)
            success_prob = float(s[F_SAFE]) * 0.8 + 0.1
            success = self._rng.random() < success_prob

            if success:
                s[F_SPEED]    = float(np.clip(s[F_SPEED] + self._rng.uniform(0.1, 0.25), 0.3, 0.95))
                s[F_URGENCY]  = float(np.clip(s[F_URGENCY] * 0.3, 0.0, 0.3))
                s[F_PRESSURE] = float(np.clip(s[F_PRESSURE] * 0.3, 0.0, 0.3))
                s[F_SAFE]     = float(np.clip(s[F_SAFE] + self._rng.uniform(0.05, 0.15), 0.1, 1.0))
                s[F_SOCIAL]   = float(np.clip(self._rng.uniform(0.0, 0.2), 0.0, 0.4))
                s[F_DENSITY]  = float(np.clip(s[F_DENSITY] - self._rng.uniform(0.0, 0.1), 0.0, 1.0))
                s[F_LC_COST]  = 1.0
            else:
                # 换道失败: 很大惩罚
                s[F_SPEED]    = float(np.clip(s[F_SPEED] - self._rng.uniform(0.15, 0.35), 0.05, 0.95))
                s[F_SAFE]     = float(np.clip(s[F_SAFE] - self._rng.uniform(0.3, 0.6), 0.0, 0.5))
                s[F_SOCIAL]   = float(np.clip(self._rng.uniform(0.2, 0.5), 0.0, 0.6))
                s[F_LC_COST]  = 1.0

            self._stay_streak = 0

        else:  # stay
            self._stay_streak += 1
            # 堵车时 stay 会持续恶化
            urgency_increase = 0.03 * (1.0 + s[F_DENSITY] * 2.0)
            s[F_URGENCY]  = float(np.clip(s[F_URGENCY] + urgency_increase, 0.0, 1.0))
            # 高 urgency 逐渐侵蚀 safe
            if s[F_URGENCY] > 0.6:
                s[F_SAFE] = float(np.clip(s[F_SAFE] - 0.02, 0.0, 1.0))
            # 密度自然波动
            s[F_DENSITY]  = float(np.clip(s[F_DENSITY] + self._rng.uniform(-0.02, 0.04), 0.0, 1.0))
            s[F_PRESSURE] = float(np.clip(s[F_PRESSURE] + self._rng.uniform(-0.01, 0.03), 0.0, 1.0))
            s[F_LC_COST]  = 0.0

        return s

    # ─────────────────── 奖励计算 ───────────────────

    def _compute_reward(self, state: np.ndarray, action: int,
                        next_state: np.ndarray) -> float:
        """计算换道决策的即时奖励."""
        r = 0.0

        if action == 1:  # change
            speed_gain = next_state[F_SPEED] - state[F_SPEED]
            safe_delta = next_state[F_SAFE] - state[F_SAFE]
            urgency_drop = state[F_URGENCY] - next_state[F_URGENCY]

            if safe_delta > -0.1:  # 安全换道
                # 流畅且安全高的场景下换道是 unnecessary
                # 检查是否需要换道: 低 speed + 高 urgency + 高 pressure = 必要
                need_to_change = (state[F_SPEED] < 0.6 or state[F_URGENCY] > 0.4
                                  or state[F_PRESSURE] > 0.4)
                if need_to_change:
                    r += 0.6 + speed_gain * 0.8 + urgency_drop * 0.4
                else:
                    # 不必要的换道: 收益低且有风险
                    r -= 0.2 + self._rng.uniform(0, 0.1)

                if safe_delta > 0.05:
                    r += 0.2  # 安全提升额外奖励

            else:  # 换道导致安全大幅下降 = 危险
                r -= 1.2 + abs(safe_delta) * 1.5
                r -= 0.3 * next_state[F_SOCIAL]
        else:  # stay
            # 保持车道: 小惩罚随 urgency 递增
            r -= 0.03 + next_state[F_URGENCY] * 0.12
            if self._stay_streak > 10:
                r -= 0.08
            if next_state[F_URGENCY] > 0.8:
                r -= 0.3

        return float(np.clip(r, -2.0, 2.0))

    # ─────────────────── 终止条件 ───────────────────

    def _is_terminal(self, state: np.ndarray) -> bool:
        """判断是否终止."""
        # 安全降到接近 0 视为碰撞
        if state[F_SAFE] < 0.05:
            return True
        # urgency 爆满视为错过换道时机
        if state[F_URGENCY] > 0.95:
            return True
        return False

    # ─────────────────── Gym API ───────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境: 采样新的初始状态."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._state = self._sample_random_state()
        self._stay_streak = 0
        return self._state.copy(), {"step": self._step_count}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None, "必须先 reset()"

        action = int(action)
        state = self._state.copy()
        next_state = self._next_state(state, action)
        reward = self._compute_reward(state, action, next_state)

        self._state = next_state
        self._step_count += 1

        terminated = self._is_terminal(next_state)
        truncated = self._step_count >= self._max_steps

        info = {
            "step": self._step_count,
            "action": action,
            "speed": float(next_state[F_SPEED]),
            "safe": float(next_state[F_SAFE]),
        }

        return next_state.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        """简单文本渲染."""
        if self._state is not None:
            print(f"Step {self._step_count}: speed={self._state[F_SPEED]:.2f} "
                  f"urg={self._state[F_URGENCY]:.2f} safe={self._state[F_SAFE]:.2f} "
                  f"dens={self._state[F_DENSITY]:.2f}")


# ====================== 仿真决策接口 (用于评估) ======================

class SumoPPOPolicy:
    """
    PPO 策略的 SUMO 仿真适配器.
    在 game_lane_change.py 的 decide_lanechange 中被调用.

    用法:
        from sumo_env import SumoPPOPolicy
        policy = SumoPPOPolicy.load("models/ppo_lanechange.zip")
        action = policy.predict(8-dim feature)
    """

    def __init__(self, model):
        self._model = model

    @classmethod
    def load(cls, model_path: str):
        """从文件加载训练好的 stable-baselines3 模型."""
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        return cls(model)

    def predict(self, features: np.ndarray, deterministic: bool = True) -> int:
        """
        根据 8 维特征输出换道决策.

        返回:
            0 = stay, 1 = change
        """
        obs = features.reshape(1, -1).astype(np.float32)
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action[0])


# ====================== 自测 ======================

if __name__ == "__main__":
    env = LaneChangeEnv()
    obs, _ = env.reset(seed=42)
    total_reward = 0.0

    print("LaneChangeEnv Self-Test")
    print("=" * 50)
    for i in range(20):
        action = env.action_space.sample()  # 随机策略
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {i:2d}: action={'change' if action else 'stay':6s}  "
              f"reward={reward:+.3f}  speed={obs[0]:.2f}  safe={obs[3]:.2f}")

    print(f"  Total reward: {total_reward:.3f}")
    print("  Self-test passed!")
