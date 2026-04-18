"""
量子电路构造模块。

本模块提供构建参数化量子电路（PQC）所需的几个核心组件：

    1. amplitude_encoding  - 把归一化的经典向量编码为量子态的振幅
    2. rotation_layer      - 每个 qubit 上一层任意单比特旋转 Rot(ω, θ, φ)
    3. entangling_layer    - 环形 CNOT / CZ 纠缠层
    4. variational_block   - rotation + entangling 的复合块（一"层"的标准单元）
    5. pqc_classifier_circuit  - 完整的分类器电路（含可选噪声）
    6. pqc_generator_circuit   - 生成式 PQC（暂留接口，便于未来扩展）

所有电路都兼容 PennyLane 的 QNode 装饰器，梯度可通过参数移位规则自动求得。

设计备注
--------
选择 Rot(ω, θ, φ) = RZ(φ) · RY(θ) · RZ(ω) 作为单比特旋转的原因:
    - 三参数覆盖了任意 SU(2) 旋转
    - PennyLane 的 parameter-shift rule 对此门有标准的两项公式
    - 与 Pennylane StronglyEntanglingLayers 的习惯一致，便于与已有 benchmark 对齐

纠缠层默认使用环形 CNOT（每个 qubit 与下一个 qubit 纠缠，最后一个绕回到第一个）。
对生成器模型替换为 CZ，使得"全 0 参数"对应恒等变换，便于生成器的稳定初始化。
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pennylane as qml

from qaaf.noise_models import NoiseChannel, NoNoise


# ---------------------------------------------------------------------------
# 编码 / 基础层
# ---------------------------------------------------------------------------

def amplitude_encoding(x: np.ndarray, wires: Sequence[int]) -> None:
    """
    将归一化经典向量 x 编码为量子态的振幅。

    Parameters
    ----------
    x : ndarray, shape (2**n,)
        已经归一化的实向量（振幅）。若未归一化，PennyLane 会自动归一化并警告。
    wires : sequence of int
        用于承载编码的 qubit 下标，长度必须 == ceil(log2(len(x)))。

    Notes
    -----
    振幅编码的优势在于用 ceil(log2 d) 个 qubit 就能表示 d 维经典数据，
    相比 angle encoding 等方案在 qubit 数量上有指数级优势。
    """
    qml.AmplitudeEmbedding(
        features=x,
        wires=wires,
        normalize=True,
        pad_with=0.0,
    )


def rotation_layer(
    params: np.ndarray,
    wires: Sequence[int],
) -> None:
    """
    对每个 qubit 施加一个任意单比特旋转 Rot(ω, θ, φ)。

    Parameters
    ----------
    params : ndarray, shape (n_qubits, 3)
        每个 qubit 的三个旋转参数。
    wires : sequence of int
    """
    assert params.shape == (len(wires), 3), (
        f"rotation_layer expects params of shape (n, 3), got {params.shape}"
    )
    for i, w in enumerate(wires):
        qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=w)


def entangling_layer(
    wires: Sequence[int],
    gate: str = "CNOT",
) -> None:
    """
    环形纠缠层。

    对 n 个 qubit 依次执行 gate(w_i, w_{i+1 mod n})，
    让每个 qubit 都与下一个 qubit 建立纠缠。

    Parameters
    ----------
    wires : sequence of int
    gate : {'CNOT', 'CZ'}
        CNOT 更强，但 CZ 在"全零参数=恒等"时更干净，适合生成器。
    """
    n = len(wires)
    if n < 2:
        return  # 单 qubit 无需纠缠
    gate_fn = {"CNOT": qml.CNOT, "CZ": qml.CZ}[gate]
    for i in range(n):
        gate_fn(wires=[wires[i], wires[(i + 1) % n]])


def variational_block(
    params: np.ndarray,
    wires: Sequence[int],
    entangle_gate: str = "CNOT",
    noise: Optional[NoiseChannel] = None,
) -> None:
    """
    "一层"标准变分块: rotation + entangle + (可选)噪声。

    Parameters
    ----------
    params : ndarray, shape (n_qubits, 3)
    wires : sequence of int
    entangle_gate : str
    noise : NoiseChannel | None
        若非 None，在本层末尾对所有 wires 施加噪声，用于含噪仿真。
    """
    rotation_layer(params, wires)
    entangling_layer(wires, gate=entangle_gate)
    if noise is not None and not isinstance(noise, NoNoise):
        noise.apply(wires)


# ---------------------------------------------------------------------------
# 分类器完整电路
# ---------------------------------------------------------------------------

def pqc_classifier_circuit(
    x: np.ndarray,
    weights: np.ndarray,
    n_data_qubits: int,
    n_anc_qubits: int,
    noise: Optional[NoiseChannel] = None,
    entangle_gate: str = "CNOT",
) -> List:
    """
    完整的分类器前向电路。

    结构:
        1. 振幅编码 x 到 data_qubits
        2. 辅助 qubit 保持在 |0⟩
        3. L 层 variational_block（覆盖 data + anc 所有 qubit）
        4. 返回辅助 qubit 上的测量概率

    Parameters
    ----------
    x : ndarray
        输入样本（已归一化）。
    weights : ndarray, shape (L, n_total, 3)
        所有变分层的参数。
    n_data_qubits : int
        编码数据用的 qubit 数，应 >= ceil(log2(len(x)))。
    n_anc_qubits : int
        辅助 qubit 数，取 ceil(log2(n_class))，用于输出概率分布。
    noise : NoiseChannel | None
    entangle_gate : str

    Returns
    -------
    list of float
        长度为 2**n_anc_qubits 的概率向量。
    """
    n_total = n_data_qubits + n_anc_qubits
    data_wires = list(range(n_data_qubits))
    anc_wires = list(range(n_data_qubits, n_total))
    all_wires = list(range(n_total))

    # 1. 振幅编码
    amplitude_encoding(x, wires=data_wires)

    # 2. 辅助 qubit 默认 |0⟩，无需额外操作

    # 3. 变分层
    n_layers = weights.shape[0]
    for layer_idx in range(n_layers):
        variational_block(
            weights[layer_idx],
            wires=all_wires,
            entangle_gate=entangle_gate,
            noise=noise,
        )

    # 4. 测量辅助 qubit
    return qml.probs(wires=anc_wires)


def build_classifier_qnode(
    n_data_qubits: int,
    n_anc_qubits: int,
    n_layers: int,
    noise: Optional[NoiseChannel] = None,
    device_name: str = "default.qubit",
    diff_method: str = "best",
    interface: str = "autograd",
    shots: Optional[int] = None,
) -> qml.QNode:
    """
    构造一个分类器 QNode 的工厂函数。

    Parameters
    ----------
    n_data_qubits : int
    n_anc_qubits : int
    n_layers : int
        变分层数。影响 weights 的形状。
    noise : NoiseChannel | None
    device_name : str
        PennyLane 设备名。含噪仿真用 'default.mixed'。
    diff_method : str
        梯度方法。"parameter-shift" 对应真实硬件；"best" / "backprop" 用于模拟加速。
    interface : str
        'autograd' / 'torch' / 'jax'。默认 'autograd' 用于 numpy 风格接口。
    shots : int | None
        测量次数；None 表示 analytic 模拟。

    Returns
    -------
    QNode
    """
    n_total = n_data_qubits + n_anc_qubits
    # 含噪仿真强制切到 default.mixed
    if noise is not None and not isinstance(noise, NoNoise) \
            and device_name == "default.qubit":
        device_name = "default.mixed"

    dev = qml.device(device_name, wires=n_total, shots=shots)

    @qml.qnode(dev, diff_method=diff_method, interface=interface)
    def circuit(x, weights):
        return pqc_classifier_circuit(
            x=x,
            weights=weights,
            n_data_qubits=n_data_qubits,
            n_anc_qubits=n_anc_qubits,
            noise=noise,
        )

    return circuit


def init_classifier_weights(
    n_layers: int,
    n_qubits: int,
    seed: Optional[int] = None,
    scale: float = 0.1,
) -> np.ndarray:
    """
    初始化分类器的变分参数。

    采用小初始化（scale=0.1）有助于避免 barren plateau 问题初期的平坦梯度。
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale * np.pi, scale * np.pi, size=(n_layers, n_qubits, 3))


# ---------------------------------------------------------------------------
# 生成器电路（为未来基于生成模型的 UAP 工作预留接口）
# ---------------------------------------------------------------------------

def pqc_generator_circuit(
    psi_in: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
) -> List:
    """
    生成器 PQC：输入一个量子态，输出一个幺正变换后的态。

    此处返回密度矩阵的对角元，便于在不含噪情况下做简化实验。
    实际使用中可根据需要换成 qml.state() 返回完整态。
    """
    wires = list(range(n_qubits))
    # 输入态制备（用 state preparation）
    qml.StatePrep(psi_in, wires=wires, normalize=True)
    # 变分部分使用 CZ 纠缠，保证全零参数=恒等
    n_layers = weights.shape[0]
    for layer_idx in range(n_layers):
        variational_block(
            weights[layer_idx], wires=wires, entangle_gate="CZ"
        )
    return qml.probs(wires=wires)


# ---------------------------------------------------------------------------
# 辅助: 参数量与电路深度
# ---------------------------------------------------------------------------

def count_parameters(n_layers: int, n_qubits: int) -> int:
    """返回分类器的总参数量 (3 * L * n)。"""
    return 3 * n_layers * n_qubits


def n_qubits_for_input_dim(d: int) -> int:
    """返回振幅编码一个 d 维向量所需的 qubit 数。"""
    return int(np.ceil(np.log2(max(d, 2))))


def n_anc_for_classes(k: int) -> int:
    """返回 k 分类问题所需的辅助 qubit 数。"""
    return max(1, int(np.ceil(np.log2(max(k, 2)))))
