"""
v2x_channel_model.py — Veins-inspired 真实 V2X 通信信道模型
=============================================================
基于 802.11p (DSRC) 物理层建模，提供距离相关 SNR → PER、
多车干扰、CSMA/CA MAC 延迟，替代原有的静态丢包率模型。

用法:
    from v2x_channel_model import V2XChannelModel
    channel = V2XChannelModel(config="urban")
    per = channel.packet_loss_rate(distance_m)
    delay = channel.mac_delay(num_contending)
    snr = channel.compute_snr(distance_m, interference_power)
"""

import math
import numpy as np
from typing import Optional


class V2XChannelModel:
    """
    802.11p 通信信道模型 (Veins 启发, 纯 Python 实现).

    主要功能:
      1. log-distance 路径损耗 + 对数正态阴影
      2. SNR → BPSK BER → PER 映射
      3. 多车同频干扰 (SINR)
      4. CSMA/CA 退避延迟
      5. 机会性连接 (距离过大时概率性断开)

    参数
    ----------
    config : str
        "highway" — 高速公路视距场景 (路径损耗指数 ~2.2)
        "urban"   — 城市多径场景    (路径损耗指数 ~2.8)
        "custom"  — 自定义参数，需传 path_loss_exponent 等
    """

    # ====================== 802.11p 物理层参数 ======================
    FREQ            = 5.890e9   # Hz, 5.89 GHz (DSRC 频段)
    TX_POWER_DBM    = 20.0      # dBm, 典型车载发射功率
    TX_GAIN_DBI     = 3.0       # dBi, 发射天线增益
    RX_GAIN_DBI     = 3.0       # dBi, 接收天线增益
    NOISE_FLOOR_DBM = -95.0     # dBm, 噪声基底 (含热噪声 + 环境噪声)
    BANDWIDTH       = 10e6      # Hz, 802.11p 信道带宽 10 MHz

    # 参考路径损耗 (自由空间 at 1m)
    REF_DIST_M      = 1.0       # m
    REF_PATH_LOSS_DB = 20.0 * math.log10(FREQ) - 147.55  # Friis 自由空间路径损耗 @1m

    # CSMA/CA 参数 (802.11p)
    SLOT_TIME       = 13e-6     # s, 时隙 13 μs
    AIFS            = 58e-6     # s, Arbitration IFS
    CW_MIN          = 3         # 最小竞争窗口
    CW_MAX          = 7         # 最大竞争窗口

    # ====================== 场景预设 ======================
    SCENARIOS = {
        "highway": {
            "path_loss_exponent": 2.2,    # 视距, 反射少
            "shadowing_std_db":   2.0,    # 阴影衰落标准差较小
        },
        "urban": {
            "path_loss_exponent": 2.8,    # 多径, 建筑遮挡
            "shadowing_std_db":   4.0,    # 阴影衰落标准差较大
        },
        "tunnel": {
            "path_loss_exponent": 3.2,    # 隧道/地下
            "shadowing_std_db":   2.5,
        },
    }

    def __init__(self, config: str = "highway",
                 path_loss_exponent: Optional[float] = None,
                 shadowing_std_db: Optional[float] = None):
        if config in self.SCENARIOS:
            params = self.SCENARIOS[config].copy()
        else:
            params = self.SCENARIOS["highway"].copy()

        self.path_loss_exponent = path_loss_exponent or params["path_loss_exponent"]
        self.shadowing_std_db = shadowing_std_db or params["shadowing_std_db"]

        # 缓存一些预计算值
        self._ref_path_loss = float(self.REF_PATH_LOSS_DB)

    # ─────────────────── 路径损耗 ───────────────────

    def _log_distance_path_loss(self, distance_m: float) -> float:
        """log-distance 路径损耗模型 (dB)."""
        if distance_m < self.REF_DIST_M:
            distance_m = self.REF_DIST_M
        return (self._ref_path_loss
                + 10.0 * self.path_loss_exponent * math.log10(distance_m)
                + np.random.normal(0.0, self.shadowing_std_db))

    # ─────────────────── SNR / SINR ───────────────────

    def compute_snr(self, distance_m: float,
                    interference_power_w: float = 0.0) -> float:
        """
        计算接收 SNR (dB).

        参数:
            distance_m: 收发距离 (m)
            interference_power_w: 干扰总功率 (W)

        返回:
            SNR (dB)
        """
        path_loss_db = self._log_distance_path_loss(distance_m)
        rx_power_dbm = (self.TX_POWER_DBM + self.TX_GAIN_DBI
                        + self.RX_GAIN_DBI - path_loss_db)

        # 噪声 + 干扰 (全转 W → dBm)
        noise_w = 10.0 ** ((self.NOISE_FLOOR_DBM - 30.0) / 10.0)  # dBm → W
        total_noise_w = noise_w + interference_power_w
        noise_dbm = 10.0 * math.log10(total_noise_w) + 30.0

        return rx_power_dbm - noise_dbm

    def compute_interference_power(self, transmitters: list,
                                   rx_pos: tuple) -> float:
        """
        计算所有并发发射机对接收机的同频干扰总功率 (W).

        参数:
            transmitters: [{"pos": (x, y)}, ...] 活性发射机列表
            rx_pos:       (x, y) 接收机位置

        返回:
            总干扰功率 (W)
        """
        total_w = 0.0
        for tx in transmitters:
            dx = tx["pos"][0] - rx_pos[0]
            dy = tx["pos"][1] - rx_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.REF_DIST_M:
                dist = self.REF_DIST_M
            path_loss_db = (self._ref_path_loss
                            + 10.0 * self.path_loss_exponent * math.log10(dist))
            rx_power_dbm = (self.TX_POWER_DBM + self.TX_GAIN_DBI
                            + self.RX_GAIN_DBI - path_loss_db)
            total_w += 10.0 ** ((rx_power_dbm - 30.0) / 10.0)
        return total_w

    # ─────────────────── 误包率 (PER) ───────────────────

    def _ber_bpsk(self, snr_db: float) -> float:
        """BPSK AWGN 信道 BER."""
        snr_linear = 10.0 ** (snr_db / 10.0)
        if snr_linear < 1e-12:
            return 0.5
        return 0.5 * math.erfc(math.sqrt(snr_linear))

    def packet_loss_rate(self, distance_m: float,
                         packet_size_bytes: int = 100,
                         interference_power_w: float = 0.0) -> float:
        """
        计算距离 distance_m 处的误包率 PER.

        参数:
            distance_m:          收发距离 (m)
            packet_size_bytes:   报文大小 (bytes)
            interference_power_w: 干扰总功率 (W)

        返回:
            PER [0, 1]
        """
        snr_db = self.compute_snr(distance_m, interference_power_w)
        ber = self._ber_bpsk(snr_db)
        per = 1.0 - (1.0 - ber) ** (packet_size_bytes * 8)
        return min(1.0, per)

    # ─────────────────── MAC 延迟 ───────────────────

    def mac_delay(self, num_contending: int = 1) -> float:
        """
        802.11p CSMA/CA 信道接入延迟 (秒).

        参数:
            num_contending: 竞争信道的节点数 (含本车)

        返回:
            随机延迟 (s)
        """
        # 竞争窗口随节点数增大而增大
        cw = min(self.CW_MIN + num_contending, self.CW_MAX + 5)
        backoff_slots = np.random.randint(0, cw + 1)
        backoff_time = backoff_slots * self.SLOT_TIME
        # AIFS + backoff + 随机处理延迟
        processing_jitter = np.random.exponential(0.0005)  # ~0.5ms
        return self.AIFS + backoff_time + processing_jitter

    # ─────────────────── 连通性 ───────────────────

    def connection_quality(self, distance_m: float,
                           packet_size_bytes: int = 100) -> float:
        """
        连接质量指数 [0, 1], 1=完美, 0=完全断开.

        用于软广播范围 (替代二值化 V2X_RANGE 判断).
        以吞吐量 1−PER 作为质量度量.
        """
        # 无干扰简化版
        snr_db = self.compute_snr(distance_m, 0.0)
        ber = self._ber_bpsk(snr_db)
        per = 1.0 - (1.0 - ber) ** (packet_size_bytes * 8)
        return float(np.clip(1.0 - per, 0.0, 1.0))

    def is_connected(self, distance_m: float,
                     threshold_per: float = 0.3,
                     packet_size_bytes: int = 100) -> bool:
        """
        判断是否连通 (PER < threshold).

        参数:
            distance_m:      收发距离 (m)
            threshold_per:   连通 PER 阈值 (默认 0.3 = 30%)
            packet_size_bytes: 报文大小
        """
        per = self.packet_loss_rate(distance_m, packet_size_bytes, 0.0)
        return per < threshold_per

    # ─────────────────── 工具方法 ───────────────────

    def get_reception_status(self, distance_m: float,
                             num_contending: int = 1,
                             packet_size_bytes: int = 100) -> dict:
        """
        一站式获取接收状态字典.

        返回:
            {
                "snr_db":           SNR (dB)
                "per":              误包率
                "packet_loss":      本次是否丢包 (bool)
                "delay_s":          MAC 延迟 (s)
                "delay_steps":      延迟等价步数 (基于 STEP_LEN=0.1s)
                "quality":          连接质量 [0,1]
                "connected":        是否连通
            }
        """
        per = self.packet_loss_rate(distance_m, packet_size_bytes)
        packet_loss = np.random.rand() < per
        delay = self.mac_delay(num_contending)
        delay_steps = max(0, int(round(delay / 0.1)))
        quality = self.connection_quality(distance_m, packet_size_bytes)
        connected = per < 0.3

        return {
            "snr_db": self.compute_snr(distance_m),
            "per": round(per, 4),
            "packet_loss": packet_loss,
            "delay_s": round(delay, 6),
            "delay_steps": delay_steps,
            "quality": round(quality, 4),
            "connected": connected,
        }


# ====================== 工厂函数 ======================

def create_v2x_channel(config: str = "highway") -> V2XChannelModel:
    """便捷工厂函数."""
    return V2XChannelModel(config)


# ====================== 自测 ======================

if __name__ == "__main__":
    channel = V2XChannelModel("highway")
    print("V2X Channel Model Self-Test")
    print("=" * 50)
    for dist in [10, 50, 100, 200, 500, 800, 1200, 1500]:
        status = channel.get_reception_status(dist, num_contending=5)
        print(f"  {dist:4d}m → SNR={status['snr_db']:5.1f}dB  "
              f"PER={status['per']:.3f}  "
              f"loss={'Y' if status['packet_loss'] else 'N'}  "
              f"delay={status['delay_s']*1000:.1f}ms  "
              f"conn={'Y' if status['connected'] else 'N'}")
