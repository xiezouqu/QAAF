"""
噪声信道模块。

本模块封装 NISQ 设备上常见的几类噪声信道，供含噪电路仿真与噪声感知攻击使用。
所有噪声模型在形式上都可以写成 Kraus 算子集合：

    rho_out = sum_k  K_k rho_in K_k^dagger

其中 sum_k K_k^dagger K_k = I 保证信道保迹。

支持的噪声类型:
    - DepolarizingNoise   - 去极化噪声 (最常用)
    - BitFlipNoise        - 比特翻转噪声
    - PhaseFlipNoise      - 相位翻转噪声
    - AmplitudeDampingNoise - 振幅阻尼 (模拟 T1 衰减)
    - CombinedNoise       - 多种噪声的组合
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class NoiseChannel(ABC):
    """
    噪声信道基类。

    子类需要实现 apply 方法，该方法在 PennyLane QNode 内部调用，按电路语义
    向指定的 wire 施加噪声。
    """

    def __init__(self, strength: float):
        assert 0.0 <= strength <= 1.0, "noise strength must be in [0, 1]"
        self.strength = float(strength)

    @abstractmethod
    def apply(self, wires):
        """向指定 wire 施加噪声。必须在 QNode 上下文内调用。"""
        ...

    @abstractmethod
    def kraus_operators(self) -> List[np.ndarray]:
        """返回 Kraus 算子列表，用于理论分析或密度矩阵仿真。"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strength={self.strength})"


# ---------------------------------------------------------------------------
# 具体噪声信道
# ---------------------------------------------------------------------------

class DepolarizingNoise(NoiseChannel):
    """
    去极化噪声。

    对单量子比特，密度矩阵以概率 p 被替换为 I/2：

        rho -> (1 - p) rho + (p / 3) (X rho X + Y rho Y + Z rho Z)

    这是 NISQ 设备上最常见的噪声模型之一，常被用作理论分析的"最坏情况"基线。
    """

    def apply(self, wires):
        for w in self._iter_wires(wires):
            qml.DepolarizingChannel(self.strength, wires=w)

    def kraus_operators(self) -> List[np.ndarray]:
        p = self.strength
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [
            np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z,
        ]

    @staticmethod
    def _iter_wires(wires):
        if isinstance(wires, int):
            yield wires
        else:
            for w in wires:
                yield w


class BitFlipNoise(NoiseChannel):
    """
    比特翻转噪声。

        rho -> (1 - p) rho + p X rho X

    对应 X 方向的随机误差，在超导量子比特中常见。
    """

    def apply(self, wires):
        for w in self._iter_wires(wires):
            qml.BitFlip(self.strength, wires=w)

    def kraus_operators(self) -> List[np.ndarray]:
        p = self.strength
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        return [np.sqrt(1 - p) * I, np.sqrt(p) * X]

    _iter_wires = staticmethod(DepolarizingNoise._iter_wires.__func__)


class PhaseFlipNoise(NoiseChannel):
    """
    相位翻转噪声。

        rho -> (1 - p) rho + p Z rho Z
    """

    def apply(self, wires):
        for w in self._iter_wires(wires):
            qml.PhaseFlip(self.strength, wires=w)

    def kraus_operators(self) -> List[np.ndarray]:
        p = self.strength
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [np.sqrt(1 - p) * I, np.sqrt(p) * Z]

    _iter_wires = staticmethod(DepolarizingNoise._iter_wires.__func__)


class AmplitudeDampingNoise(NoiseChannel):
    """
    振幅阻尼噪声。

    这类噪声非对称，建模的是 |1⟩ -> |0⟩ 的自发辐射过程（T1 衰减）。
    Kraus 算子为:
        K0 = diag(1, sqrt(1-gamma))
        K1 = [[0, sqrt(gamma)], [0, 0]]
    """

    def apply(self, wires):
        for w in self._iter_wires(wires):
            qml.AmplitudeDamping(self.strength, wires=w)

    def kraus_operators(self) -> List[np.ndarray]:
        g = self.strength
        K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
        return [K0, K1]

    _iter_wires = staticmethod(DepolarizingNoise._iter_wires.__func__)


# ---------------------------------------------------------------------------
# 组合与 null 噪声
# ---------------------------------------------------------------------------

class NoNoise(NoiseChannel):
    """用作"无噪声"占位符，便于在同一代码路径中开关噪声。"""

    def __init__(self):
        super().__init__(strength=0.0)

    def apply(self, wires):
        return  # 不做任何操作

    def kraus_operators(self) -> List[np.ndarray]:
        return [np.eye(2, dtype=complex)]


class CombinedNoise(NoiseChannel):
    """
    多种噪声的顺序组合。

    实际 NISQ 设备常常同时存在多种噪声源，这个类允许把几个 NoiseChannel
    串在一起作用到同一组 wires 上。

    Example
    -------
    >>> noise = CombinedNoise([
    ...     DepolarizingNoise(0.01),
    ...     AmplitudeDampingNoise(0.005),
    ... ])
    """

    def __init__(self, channels: List[NoiseChannel]):
        super().__init__(strength=max(c.strength for c in channels))
        self.channels = channels

    def apply(self, wires):
        for ch in self.channels:
            ch.apply(wires)

    def kraus_operators(self) -> List[np.ndarray]:
        # 组合信道的 Kraus 算子是各信道 Kraus 算子的张量乘法展开，此处仅返回
        # 第一个信道的 Kraus 作为近似，用于需要单一信道表示的分析场景。
        # 注: 严格组合请在外部用密度矩阵演化处理。
        return self.channels[0].kraus_operators()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """噪声配置容器，便于从 YAML / dict 构造。"""
    type: str = "none"
    strength: float = 0.0
    extra: Optional[dict] = None


def build_noise(config: NoiseConfig | dict | None) -> NoiseChannel:
    """
    从配置构造噪声信道。

    Parameters
    ----------
    config : NoiseConfig | dict | None
        噪声配置。如果是 None，返回 NoNoise。

    Returns
    -------
    NoiseChannel
    """
    if config is None:
        return NoNoise()
    if isinstance(config, dict):
        config = NoiseConfig(**config)

    t = config.type.lower()
    s = config.strength

    registry = {
        "none": lambda: NoNoise(),
        "depolarizing": lambda: DepolarizingNoise(s),
        "bit_flip": lambda: BitFlipNoise(s),
        "phase_flip": lambda: PhaseFlipNoise(s),
        "amplitude_damping": lambda: AmplitudeDampingNoise(s),
    }
    if t not in registry:
        raise ValueError(
            f"Unknown noise type: {t}. Available: {list(registry.keys())}"
        )
    return registry[t]()
