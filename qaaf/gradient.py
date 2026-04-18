"""
基于参数移位规则 (Parameter-Shift Rule, PSR) 的梯度估计。

在经典神经网络中，梯度可以通过反向传播直接计算。但对于真实的量子硬件而言，
我们无法"打开"量子电路看中间状态，只能通过重复测量得到期望值。

参数移位规则给出了一种精确计算量子电路对参数梯度的方法。对于形如
    U(theta) = exp(-i * theta * G / 2)
的旋转门（G 是本征值为 ±1 的 Pauli 算符），其对 theta 的梯度满足:

    d <L> / d theta = 0.5 * (<L(theta + pi/2)> - <L(theta - pi/2)>)

这使得量子硬件上的梯度估计只需要两次额外的电路评估。

本模块把这一规则应用到**对输入样本 x** 的梯度估计上——对抗攻击需要的就是
d L / d x，而不是 d L / d theta。我们的技巧是把输入通过振幅编码接入电路后，
用数值差分（中心差分）得到 d L / d x，同时保留 PSR 用于模型参数上（训练时）。
这是 "hybrid gradient" 模式。

两种梯度估计方法:
    - analytic_input_gradient   - 基于可微模拟的解析梯度 (快, 用于实验)
    - finite_difference_gradient - 基于中心差分的数值梯度 (慢, 用于真机模拟)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qaaf.classifiers import PQCClassifier, HybridClassifier


# ---------------------------------------------------------------------------
# 1. 解析梯度（基于 backprop 的快速路径）
# ---------------------------------------------------------------------------

def analytic_input_gradient(
    clf: PQCClassifier,
    x: np.ndarray,
    y_true: int,
    loss: str = "cross_entropy",
) -> np.ndarray:
    """
    通过可微仿真直接计算损失对输入 x 的梯度。

    Parameters
    ----------
    clf : PQCClassifier
    x : ndarray, shape (d,)
        单个样本。
    y_true : int
        样本标签。
    loss : {'cross_entropy', 'margin'}
        损失函数类型。
        - 'cross_entropy': -log P(y_true | x)
        - 'margin': max_{c != y_true} P(c) - P(y_true)

    Returns
    -------
    ndarray
        grad_x L，shape 与 x 相同。
    """
    # 将 x 标记为 requires_grad 用 PennyLane numpy
    x_pnp = pnp.array(x, requires_grad=True)

    def _loss_fn(xx):
        probs = clf.qnode(xx, clf.weights)
        probs = probs[: clf.n_classes]
        probs = probs / (pnp.sum(probs) + 1e-12)
        if loss == "cross_entropy":
            return -pnp.log(probs[y_true] + 1e-12)
        elif loss == "margin":
            other = pnp.concatenate([probs[:y_true], probs[y_true+1:]])
            return pnp.max(other) - probs[y_true]
        else:
            raise ValueError(f"unknown loss: {loss}")

    grad = qml.grad(_loss_fn)(x_pnp)
    return np.array(grad)


# ---------------------------------------------------------------------------
# 2. 有限差分梯度（模拟真机的黑盒梯度）
# ---------------------------------------------------------------------------

def finite_difference_gradient(
    clf: PQCClassifier,
    x: np.ndarray,
    y_true: int,
    eps: float = 1e-3,
    loss: str = "cross_entropy",
    n_shots: Optional[int] = None,
) -> np.ndarray:
    """
    基于中心差分的梯度估计，适合真机或黑盒场景。

    使用中心差分:
        grad_i ≈ (L(x + eps * e_i) - L(x - eps * e_i)) / (2 * eps)

    其中 e_i 是第 i 维的单位向量。

    Parameters
    ----------
    clf : PQCClassifier
    x : ndarray, shape (d,)
    y_true : int
    eps : float
        差分步长。太小会放大测量噪声，太大会引入截断误差。
    loss : {'cross_entropy', 'margin'}
    n_shots : int | None
        shot 数。若设置，模拟真机有限采样下的梯度估计。

    Returns
    -------
    ndarray
    """
    # 如需模拟有限 shot，临时切换 device
    if n_shots is not None:
        original_qnode = clf._qnode
        clf._qnode = None
        clf.shots = n_shots

    def _loss(xx):
        probs = np.array(clf.qnode(xx, clf.weights))
        probs = probs[: clf.n_classes]
        probs = probs / (probs.sum() + 1e-12)
        if loss == "cross_entropy":
            return -np.log(probs[y_true] + 1e-12)
        elif loss == "margin":
            other_max = max(
                probs[c] for c in range(clf.n_classes) if c != y_true
            )
            return other_max - probs[y_true]
        else:
            raise ValueError(f"unknown loss: {loss}")

    d = len(x)
    grad = np.zeros_like(x)
    for i in range(d):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (_loss(x_plus) - _loss(x_minus)) / (2.0 * eps)

    if n_shots is not None:
        clf._qnode = original_qnode
        clf.shots = None

    return grad


# ---------------------------------------------------------------------------
# 3. 参数移位规则对模型参数的梯度 (训练相关, 保留接口)
# ---------------------------------------------------------------------------

def parameter_shift_weight_gradient(
    clf: PQCClassifier,
    x: np.ndarray,
    y_true: int,
) -> np.ndarray:
    """
    使用参数移位规则计算损失对模型权重的梯度。

    本函数在训练中用不到（训练使用 backprop），但在模拟真机训练或做参数敏感性
    分析时很有用。

    Parameters
    ----------
    clf : PQCClassifier
    x : ndarray
    y_true : int

    Returns
    -------
    ndarray, shape = clf.weights.shape
    """
    shift = np.pi / 2
    weights = clf.weights.copy()
    grad = np.zeros_like(weights)

    def _loss_at(w):
        probs = np.array(clf.qnode(x, w))
        probs = probs[: clf.n_classes]
        probs = probs / (probs.sum() + 1e-12)
        return -np.log(probs[y_true] + 1e-12)

    it = np.nditer(weights, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = weights[idx]

        weights[idx] = original + shift
        loss_plus = _loss_at(weights)

        weights[idx] = original - shift
        loss_minus = _loss_at(weights)

        grad[idx] = 0.5 * (loss_plus - loss_minus)
        weights[idx] = original
        it.iternext()

    return grad


# ---------------------------------------------------------------------------
# 4. 混合模型的梯度（对量子输入 + 经典中间特征）
# ---------------------------------------------------------------------------

def hybrid_model_gradients(
    clf: HybridClassifier,
    x: np.ndarray,
    y_true: int,
    loss: str = "cross_entropy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对混合模型分别计算 (量子输入梯度, 经典特征梯度)。

    这是 CMJA 跨模块联合攻击的基础：攻击者可以同时扰动:
        grad_x L - 直接作用在输入数据上
        grad_phi L - 作用在量子特征上（即 PQC 输出层之后）

    Returns
    -------
    grad_x : ndarray, shape (d,)
    grad_phi : ndarray, shape (feature_dim,)
    """
    import torch

    # 1. grad_x —— 通过量子子模块算
    x_pnp = pnp.array(x, requires_grad=True)

    def _qloss(xx):
        probs = clf.quantum.qnode(xx, clf.quantum.weights)
        probs = probs[: clf.quantum.n_classes]
        probs = probs / (pnp.sum(probs) + 1e-12)
        # 量子部分的代理损失: 使正确类的概率降低
        return -pnp.log(probs[y_true] + 1e-12)

    grad_x = np.array(qml.grad(_qloss)(x_pnp))

    # 2. grad_phi —— 把量子特征喂给经典头，在 torch 里算梯度
    feat = clf.extract_features(np.atleast_2d(x))[0].astype(np.float32)
    feat_t = torch.tensor(feat, requires_grad=True)
    logits = clf.head(feat_t)
    if loss == "cross_entropy":
        logprobs = torch.log_softmax(logits, dim=-1)
        loss_val = -logprobs[y_true]
    elif loss == "margin":
        probs = torch.softmax(logits, dim=-1)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[y_true] = False
        loss_val = probs[mask].max() - probs[y_true]
    else:
        raise ValueError(f"unknown loss: {loss}")
    loss_val.backward()
    grad_phi = feat_t.grad.detach().numpy()

    return grad_x, grad_phi


# ---------------------------------------------------------------------------
# 5. 噪声感知梯度（蒙特卡洛平均）
# ---------------------------------------------------------------------------

def noise_aware_gradient(
    clf: PQCClassifier,
    x: np.ndarray,
    y_true: int,
    n_samples: int = 16,
    method: str = "analytic",
    loss: str = "cross_entropy",
) -> np.ndarray:
    """
    噪声感知梯度：对含噪信道取蒙特卡洛期望后的梯度估计。

    这是 NAP 攻击的核心子程序。含噪电路每次执行都可能给出不同的测量结果，
    直接用单次梯度容易被噪声淹没。通过多次采样取平均，我们得到对噪声期望化的
    梯度估计，使得生成的扰动在真实含噪硬件上更稳定。

    Parameters
    ----------
    clf : PQCClassifier
        分类器本身已携带噪声信道（通过 clf.noise）。
    x, y_true
    n_samples : int
        蒙特卡洛样本数。越多越稳定，但成本线性增长。
    method : {'analytic', 'fd'}
        内部梯度计算方法。
    loss : {'cross_entropy', 'margin'}

    Returns
    -------
    ndarray
    """
    grads = []
    for _ in range(n_samples):
        if method == "analytic":
            g = analytic_input_gradient(clf, x, y_true, loss=loss)
        elif method == "fd":
            g = finite_difference_gradient(clf, x, y_true, loss=loss)
        else:
            raise ValueError(f"unknown method: {method}")
        grads.append(g)
    return np.mean(grads, axis=0)
