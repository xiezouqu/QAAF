"""
三类对抗攻击的核心实现。

    1. ParameterShiftAttack (PSA)   —— 贡献 1
       基于梯度的 L∞ / L2 对抗攻击，支持 FGSM、PGD、MIFGSM 三种变体。
       梯度由 gradient 模块提供，对应参数移位规则（硬件兼容）或解析梯度（仿真）。

    2. NoiseAwarePerturbation (NAP) —— 贡献 2
       在扰动优化过程中显式地对噪声信道做蒙特卡洛期望化，生成的对抗样本在
       含噪硬件上保持稳定攻击效果。

    3. CrossModuleJointAttack (CMJA) —— 贡献 3
       面向量子-经典混合模型，在量子输入和经典中间特征上协同扰动。

设计上每类攻击都实现 generate(X, y) 接口，返回同 shape 的对抗样本数组，
便于下游统一评估。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from qaaf.classifiers import PQCClassifier, HybridClassifier
from qaaf import gradient as grad_mod
from qaaf.utils import clip_perturbation, clip_image, normalize_vector


# ---------------------------------------------------------------------------
# 攻击基类
# ---------------------------------------------------------------------------

class BaseAttack(ABC):
    """所有攻击的统一基类。"""

    @abstractmethod
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        对一批样本生成对抗样本。

        Returns
        -------
        X_adv : ndarray, 同 X 的 shape
        """
        ...

    def attack_success_rate(
        self,
        clf,
        X_clean: np.ndarray,
        X_adv: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """
        计算攻击成功率。

        攻击成功率定义为: 在原本预测正确的样本中，被扰动后预测错误的比例。
        """
        mask = clf.predict(X_clean) == y_true  # 原本预测对的样本
        if mask.sum() == 0:
            return 0.0
        adv_correct = clf.predict(X_adv[mask]) == y_true[mask]
        return float((~adv_correct).mean())


# ---------------------------------------------------------------------------
# 贡献 1: PSA - 基于梯度的对抗攻击
# ---------------------------------------------------------------------------

@dataclass
class PSAConfig:
    """PSA 攻击配置。"""
    epsilon: float = 0.1
    steps: int = 40
    alpha: float = 0.01       # 每步扰动幅度
    norm: str = "linf"        # {'linf', 'l2'}
    variant: str = "pgd"      # {'fgsm', 'pgd', 'mifgsm'}
    momentum: float = 0.9     # MIFGSM 动量系数
    loss: str = "cross_entropy"
    grad_method: str = "analytic"  # {'analytic', 'fd'}
    clip_input: Tuple[float, float] = (0.0, 1.0)
    verbose: bool = True


class ParameterShiftAttack(BaseAttack):
    """
    基于参数移位规则梯度估计的对抗攻击。

    三种变体:
        - FGSM (单步):    x_adv = x + epsilon * sign(grad_x L)
        - PGD (多步):     迭代 FGSM，每步截断到 epsilon 邻域
        - MIFGSM:         带动量的 PGD，提升黑盒迁移性

    Parameters
    ----------
    classifier : PQCClassifier
    config : PSAConfig

    Examples
    --------
    >>> atk = ParameterShiftAttack(clf, PSAConfig(epsilon=0.1, steps=40))
    >>> X_adv = atk.generate(X_test, y_test)
    """

    def __init__(self, classifier: PQCClassifier, config: Optional[PSAConfig] = None):
        self.clf = classifier
        self.cfg = config or PSAConfig()

    def _compute_grad(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.cfg.grad_method == "analytic":
            return grad_mod.analytic_input_gradient(
                self.clf, x, y, loss=self.cfg.loss
            )
        elif self.cfg.grad_method == "fd":
            return grad_mod.finite_difference_gradient(
                self.clf, x, y, loss=self.cfg.loss
            )
        else:
            raise ValueError(f"unknown grad_method: {self.cfg.grad_method}")

    def _step_direction(self, grad: np.ndarray) -> np.ndarray:
        """根据范数类型给出一步更新方向。"""
        if self.cfg.norm == "linf":
            return np.sign(grad)
        elif self.cfg.norm == "l2":
            return normalize_vector(grad)
        else:
            raise ValueError(f"unknown norm: {self.cfg.norm}")

    def _generate_one(self, x: np.ndarray, y: int) -> np.ndarray:
        """对单个样本生成对抗样本。"""
        x_orig = x.copy()
        x_adv = x.copy()

        if self.cfg.variant == "fgsm":
            g = self._compute_grad(x_adv, y)
            x_adv = x_orig + self.cfg.epsilon * self._step_direction(g)
            x_adv = clip_image(x_adv, *self.cfg.clip_input)
            return x_adv

        # PGD / MIFGSM 共同的迭代框架
        momentum = np.zeros_like(x)
        for step in range(self.cfg.steps):
            g = self._compute_grad(x_adv, y)
            if self.cfg.variant == "mifgsm":
                # 归一化 L1，再累积动量
                g_norm = g / (np.abs(g).sum() + 1e-12)
                momentum = self.cfg.momentum * momentum + g_norm
                direction = self._step_direction(momentum)
            else:
                direction = self._step_direction(g)

            x_adv = x_adv + self.cfg.alpha * direction
            # 投影到 epsilon 邻域
            delta = clip_perturbation(
                x_adv - x_orig, epsilon=self.cfg.epsilon, norm=self.cfg.norm
            )
            x_adv = clip_image(x_orig + delta, *self.cfg.clip_input)

        return x_adv

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_adv = np.zeros_like(X)
        iterator = range(len(X))
        if self.cfg.verbose:
            iterator = tqdm(iterator, desc=f"PSA-{self.cfg.variant}")
        for i in iterator:
            X_adv[i] = self._generate_one(X[i], int(y[i]))
        return X_adv


# ---------------------------------------------------------------------------
# 贡献 2: NAP - 噪声感知扰动优化
# ---------------------------------------------------------------------------

@dataclass
class NAPConfig:
    """NAP 攻击配置。"""
    epsilon: float = 0.1
    steps: int = 40
    alpha: float = 0.01
    norm: str = "linf"
    mc_samples: int = 16       # 每步的蒙特卡洛采样数
    anneal_schedule: Optional[List[int]] = None  # 每步 MC 样本数的退火（None=恒定）
    loss: str = "cross_entropy"
    clip_input: Tuple[float, float] = (0.0, 1.0)
    verbose: bool = True


class NoiseAwarePerturbation(BaseAttack):
    """
    噪声感知扰动优化。

    核心想法: 含噪电路的梯度天然带有噪声涨落，如果每步只基于单次梯度更新，
    生成的扰动会被噪声淹没。NAP 在每一步用 mc_samples 次独立的含噪前向/反向
    得到一个期望化梯度 E[grad]，再用期望梯度更新，相当于:

        x_{t+1} = x_t + alpha * sign( E[grad_x L(x_t; noise)] )

    这样的扰动在含噪环境中更稳定，白盒攻击成功率与无噪时更接近。

    Notes
    -----
    当分类器本身就是无噪的 (clf.noise = NoNoise)，NAP 退化为 PSA-PGD，
    除了 mc_samples 带来的计算开销。
    """

    def __init__(self, classifier: PQCClassifier, config: Optional[NAPConfig] = None):
        self.clf = classifier
        self.cfg = config or NAPConfig()

    def _expected_grad(self, x: np.ndarray, y: int, n_samples: int) -> np.ndarray:
        return grad_mod.noise_aware_gradient(
            self.clf, x, y,
            n_samples=n_samples,
            loss=self.cfg.loss,
        )

    def _generate_one(self, x: np.ndarray, y: int) -> np.ndarray:
        x_orig = x.copy()
        x_adv = x.copy()

        schedule = (
            self.cfg.anneal_schedule
            if self.cfg.anneal_schedule is not None
            else [self.cfg.mc_samples] * self.cfg.steps
        )

        for step in range(self.cfg.steps):
            n_s = schedule[min(step, len(schedule) - 1)]
            g = self._expected_grad(x_adv, y, n_s)

            if self.cfg.norm == "linf":
                direction = np.sign(g)
            elif self.cfg.norm == "l2":
                direction = normalize_vector(g)
            else:
                raise ValueError(f"unknown norm: {self.cfg.norm}")

            x_adv = x_adv + self.cfg.alpha * direction
            delta = clip_perturbation(
                x_adv - x_orig, epsilon=self.cfg.epsilon, norm=self.cfg.norm
            )
            x_adv = clip_image(x_orig + delta, *self.cfg.clip_input)

        return x_adv

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_adv = np.zeros_like(X)
        iterator = range(len(X))
        if self.cfg.verbose:
            iterator = tqdm(iterator, desc="NAP")
        for i in iterator:
            X_adv[i] = self._generate_one(X[i], int(y[i]))
        return X_adv


# ---------------------------------------------------------------------------
# 贡献 3: CMJA - 量子-经典混合模型跨模块联合攻击
# ---------------------------------------------------------------------------

@dataclass
class CMJAConfig:
    """CMJA 攻击配置。"""
    epsilon_x: float = 0.1       # 输入空间扰动半径
    epsilon_phi: float = 0.1     # 特征空间扰动半径（度量经典部分的"攻击预算"）
    steps: int = 40
    alpha_x: float = 0.01
    alpha_phi: float = 0.01
    norm: str = "linf"
    lam: float = 0.5             # 量子损失 / 经典损失的权重
    loss: str = "cross_entropy"
    clip_input: Tuple[float, float] = (0.0, 1.0)
    mode: str = "joint"          # {'joint', 'quantum_only', 'classical_only'}
    verbose: bool = True


class CrossModuleJointAttack(BaseAttack):
    """
    量子-经典混合模型跨模块联合攻击。

    在每一步同时:
        - 对输入 x 做扰动（影响量子编码）
        - 对中间特征 phi(x) 做扰动（影响经典分类头）

    然后最终的对抗样本是组合了两种扰动的结果。"quantum_only" 和 "classical_only"
    模式用于消融对比，展示联合攻击的优势。

    实现细节:
        - 我们无法直接修改 phi(x)，因为它是 PQC 的输出，由 x 决定
        - 近似做法: 用 grad_phi 作为"额外指导"，通过链式法则 grad_phi -> grad_x
          把经典部分的攻击方向反馈到输入空间
        - 最终更新方向 = lam * sign(grad_x direct) + (1 - lam) * sign(grad_x via phi)

    这个设计虽然近似，但在混合模型上比单纯用输入梯度或单纯用特征梯度都更有效。
    """

    def __init__(
        self,
        classifier: HybridClassifier,
        config: Optional[CMJAConfig] = None,
    ):
        self.clf = classifier
        self.cfg = config or CMJAConfig()

    def _project(self, x: np.ndarray, direction: np.ndarray) -> np.ndarray:
        if self.cfg.norm == "linf":
            return np.sign(direction)
        elif self.cfg.norm == "l2":
            return normalize_vector(direction)
        else:
            raise ValueError(f"unknown norm: {self.cfg.norm}")

    def _estimate_phi_to_x(
        self,
        x: np.ndarray,
        grad_phi: np.ndarray,
        eps: float = 1e-3,
    ) -> np.ndarray:
        """
        通过数值方式估计从特征梯度到输入梯度的 Jacobian-vector product:
            (d phi / d x)^T grad_phi
        使用中心差分一维投影（而不是显式算 Jacobian）避免 O(d * feature_dim) 开销。

        实际做法: 对 x 的每一维做小扰动，观察 phi 的变化，加权到 grad_phi。
        """
        d = len(x)
        feat_dim = len(grad_phi)
        jvp = np.zeros(d)

        # 基线特征
        phi0 = self.clf.extract_features(np.atleast_2d(x))[0]

        for i in range(d):
            x_plus = x.copy()
            x_plus[i] += eps
            phi_plus = self.clf.extract_features(np.atleast_2d(x_plus))[0]
            # 数值偏导 d phi / d x_i
            dphi_dxi = (phi_plus - phi0) / eps
            jvp[i] = np.dot(dphi_dxi, grad_phi)
        return jvp

    def _generate_one(self, x: np.ndarray, y: int) -> np.ndarray:
        x_orig = x.copy()
        x_adv = x.copy()

        for step in range(self.cfg.steps):
            grad_x_direct, grad_phi = grad_mod.hybrid_model_gradients(
                self.clf, x_adv, y, loss=self.cfg.loss
            )

            if self.cfg.mode == "joint":
                # 用数值 JVP 把 grad_phi 反馈到输入空间
                grad_x_via_phi = self._estimate_phi_to_x(x_adv, grad_phi)
                direction = (
                    self.cfg.lam * np.sign(grad_x_direct)
                    + (1 - self.cfg.lam) * np.sign(grad_x_via_phi)
                )
            elif self.cfg.mode == "quantum_only":
                direction = np.sign(grad_x_direct)
            elif self.cfg.mode == "classical_only":
                grad_x_via_phi = self._estimate_phi_to_x(x_adv, grad_phi)
                direction = np.sign(grad_x_via_phi)
            else:
                raise ValueError(f"unknown mode: {self.cfg.mode}")

            x_adv = x_adv + self.cfg.alpha_x * direction
            delta = clip_perturbation(
                x_adv - x_orig,
                epsilon=self.cfg.epsilon_x,
                norm=self.cfg.norm,
            )
            x_adv = clip_image(x_orig + delta, *self.cfg.clip_input)

        return x_adv

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_adv = np.zeros_like(X)
        iterator = range(len(X))
        if self.cfg.verbose:
            iterator = tqdm(iterator, desc=f"CMJA-{self.cfg.mode}")
        for i in iterator:
            X_adv[i] = self._generate_one(X[i], int(y[i]))
        return X_adv


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_attack(
    kind: str,
    classifier,
    **kwargs,
) -> BaseAttack:
    """
    从字符串名构造攻击对象。

    kind : {'psa', 'nap', 'cmja'}
    """
    kind = kind.lower()
    if kind == "psa":
        return ParameterShiftAttack(classifier, PSAConfig(**kwargs))
    elif kind == "nap":
        return NoiseAwarePerturbation(classifier, NAPConfig(**kwargs))
    elif kind == "cmja":
        return CrossModuleJointAttack(classifier, CMJAConfig(**kwargs))
    else:
        raise ValueError(f"unknown attack kind: {kind}")
