"""
量子分类器的高层包装。

本模块把 circuits 模块里的 QNode 包装成 sklearn 风格的分类器类，
对外提供统一的 fit / predict / predict_proba 接口。

提供三种分类器:
    PQCClassifier        - 纯量子分类器（一个 PQC 从输入到输出概率）
    HybridClassifier     - 量子-经典混合分类器（PQC 提取特征 + 经典分类头）
    QuantumEnsemble      - 多个 PQC 投票集成（用作 baseline）

核心设计:
    - fit 使用 numpy + PennyLane autograd 做梯度下降
    - 支持 mini-batch、学习率衰减、早停
    - 保存/加载 pickle 格式的权重
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qaaf import circuits
from qaaf.noise_models import NoiseChannel, NoNoise
from qaaf.utils import get_logger, save_pickle, load_pickle


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """分类器训练的超参数。"""
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.01
    lr_decay: float = 0.5      # 每过 lr_decay_epochs 乘一次
    lr_decay_epochs: int = 10
    early_stop_patience: int = 5
    verbose: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# 纯量子分类器
# ---------------------------------------------------------------------------

class PQCClassifier:
    """
    纯 PQC 分类器。

    Parameters
    ----------
    n_data_qubits : int
    n_anc_qubits : int
    n_layers : int
    noise : NoiseChannel | None
    diff_method : str
        PennyLane 梯度方法。训练时推荐 'best'（自动选 backprop）加速，
        attack / inference 阶段可以切到 'parameter-shift' 模拟硬件。

    Examples
    --------
    >>> clf = PQCClassifier(n_data_qubits=8, n_anc_qubits=1, n_layers=10)
    >>> clf.fit(X_train, y_train, TrainConfig(epochs=20))
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_data_qubits: int,
        n_anc_qubits: int = 1,
        n_layers: int = 10,
        noise: Optional[NoiseChannel] = None,
        diff_method: str = "best",
        shots: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_data_qubits = n_data_qubits
        self.n_anc_qubits = n_anc_qubits
        self.n_layers = n_layers
        self.n_qubits_total = n_data_qubits + n_anc_qubits
        self.noise = noise if noise is not None else NoNoise()
        self.diff_method = diff_method
        self.shots = shots
        self.seed = seed

        # 延迟构造 QNode（fit 或 predict 时首次使用）
        self._qnode: Optional[qml.QNode] = None
        self.weights = circuits.init_classifier_weights(
            n_layers=n_layers,
            n_qubits=self.n_qubits_total,
            seed=seed,
        )
        self.n_classes: Optional[int] = None
        self.logger = get_logger("PQCClassifier")
        self.history: Dict[str, List[float]] = {"loss": [], "acc": []}

    # ------------------------------------------------------------------
    # QNode 构造 / 缓存
    # ------------------------------------------------------------------

    def _build_qnode(self, interface: str = "autograd") -> qml.QNode:
        return circuits.build_classifier_qnode(
            n_data_qubits=self.n_data_qubits,
            n_anc_qubits=self.n_anc_qubits,
            n_layers=self.n_layers,
            noise=self.noise,
            diff_method=self.diff_method,
            interface=interface,
            shots=self.shots,
        )

    @property
    def qnode(self) -> qml.QNode:
        if self._qnode is None:
            self._qnode = self._build_qnode()
        return self._qnode

    # ------------------------------------------------------------------
    # 前向 / 预测
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的类概率分布。"""
        X = np.atleast_2d(X)
        probs = np.array([self.qnode(x, self.weights) for x in X])
        # 只保留前 n_classes 个类别对应的概率（忽略 padding）
        if self.n_classes is not None:
            probs = probs[:, : self.n_classes]
            probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _loss_fn(self, weights, x, y_onehot):
        probs = self.qnode(x, weights)
        # 只取前 n_classes 个位置
        probs = probs[: self.n_classes]
        probs = probs / (pnp.sum(probs) + 1e-12)
        # 交叉熵
        return -pnp.sum(y_onehot * pnp.log(probs + 1e-12))

    def _batch_loss(self, weights, X_batch, Y_batch):
        total = 0.0
        for x, y in zip(X_batch, Y_batch):
            total = total + self._loss_fn(weights, x, y)
        return total / len(X_batch)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[TrainConfig] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "PQCClassifier":
        """
        训练分类器。

        使用 mini-batch SGD + 可选学习率衰减与早停。
        """
        config = config or TrainConfig()
        self.n_classes = int(y.max()) + 1
        assert self.n_classes <= 2 ** self.n_anc_qubits, (
            f"n_classes={self.n_classes} exceeds capacity of "
            f"n_anc_qubits={self.n_anc_qubits} (max={2 ** self.n_anc_qubits})"
        )

        # 转 one-hot
        Y_oh = np.eye(self.n_classes)[y]

        # 用 PennyLane numpy 包装权重，开启梯度
        weights = pnp.array(self.weights, requires_grad=True)

        # 选优化器
        optimizer = qml.AdamOptimizer(stepsize=config.lr)

        n_samples = len(X)
        rng = np.random.default_rng(config.seed)
        best_val = -np.inf
        patience = 0

        for epoch in range(config.epochs):
            # 学习率衰减
            if (epoch > 0) and (epoch % config.lr_decay_epochs == 0):
                optimizer.stepsize *= config.lr_decay
                if config.verbose:
                    self.logger.info(
                        f"epoch {epoch}: lr decayed to {optimizer.stepsize:.5f}"
                    )

            # 打乱 + 分 batch
            idx = rng.permutation(n_samples)
            losses = []
            for start in range(0, n_samples, config.batch_size):
                batch_idx = idx[start : start + config.batch_size]
                Xb, Yb = X[batch_idx], Y_oh[batch_idx]

                def cost(w):
                    return self._batch_loss(w, Xb, Yb)

                weights, loss_val = optimizer.step_and_cost(cost, weights)
                losses.append(float(loss_val))

            # 同步回 self.weights（断开梯度）
            self.weights = np.array(weights)
            mean_loss = float(np.mean(losses))
            train_acc = self.score(X, y)
            self.history["loss"].append(mean_loss)
            self.history["acc"].append(train_acc)

            val_msg = ""
            if X_val is not None:
                val_acc = self.score(X_val, y_val)
                val_msg = f", val_acc={val_acc:.4f}"
                if val_acc > best_val:
                    best_val = val_acc
                    patience = 0
                else:
                    patience += 1
                    if patience >= config.early_stop_patience:
                        if config.verbose:
                            self.logger.info(
                                f"Early stopping at epoch {epoch}"
                            )
                        break

            if config.verbose:
                self.logger.info(
                    f"epoch {epoch+1}/{config.epochs} "
                    f"loss={mean_loss:.4f} train_acc={train_acc:.4f}{val_msg}"
                )

        return self

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存模型权重和超参数到 pickle 文件。"""
        state = {
            "weights": self.weights,
            "n_data_qubits": self.n_data_qubits,
            "n_anc_qubits": self.n_anc_qubits,
            "n_layers": self.n_layers,
            "n_classes": self.n_classes,
            "history": self.history,
            "seed": self.seed,
        }
        save_pickle(state, path)

    @classmethod
    def load(cls, path: str, noise: Optional[NoiseChannel] = None) -> "PQCClassifier":
        """从文件加载。"""
        state = load_pickle(path)
        clf = cls(
            n_data_qubits=state["n_data_qubits"],
            n_anc_qubits=state["n_anc_qubits"],
            n_layers=state["n_layers"],
            noise=noise,
            seed=state["seed"],
        )
        clf.weights = state["weights"]
        clf.n_classes = state["n_classes"]
        clf.history = state.get("history", {"loss": [], "acc": []})
        return clf


# ---------------------------------------------------------------------------
# 混合分类器：PQC 特征 + 经典分类头
# ---------------------------------------------------------------------------

class HybridClassifier:
    """
    量子-经典混合分类器。

    架构:
        x -> [PQC feature extractor] -> phi(x) -> [classical MLP head] -> logits

    PQC 部分只做特征提取（返回辅助 qubit 概率作为 "量子特征"），
    经典 MLP 作为分类头。这是实际 NISQ 部署中常见的设计，因为经典头可以很廉价地
    把量子特征映射到任意多类上，而不需要扩展辅助 qubit。

    本类的关键用途是支持 CMJA 跨模块联合攻击：攻击者可以同时扰动
    量子部分的输入 x 和经典部分的中间特征 phi(x)。
    """

    def __init__(
        self,
        n_data_qubits: int,
        n_anc_qubits: int = 2,
        n_layers: int = 6,
        n_classes: int = 2,
        head_hidden: int = 16,
        noise: Optional[NoiseChannel] = None,
        seed: int = 42,
    ):
        # 量子特征提取器
        self.quantum = PQCClassifier(
            n_data_qubits=n_data_qubits,
            n_anc_qubits=n_anc_qubits,
            n_layers=n_layers,
            noise=noise,
            seed=seed,
        )
        # 经典分类头（延迟导入 torch，保持模块级依赖最小）
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.n_classes = n_classes
        self.feature_dim = 2 ** n_anc_qubits

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_classes),
        )

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """提取量子特征 phi(x)。"""
        return self.quantum.predict_proba(X)  # 使用辅助 qubit 概率作为量子特征

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = self.extract_features(X)
        with self.torch.no_grad():
            logits = self.head(self.torch.from_numpy(feats.astype(np.float32)))
            probs = self.torch.softmax(logits, dim=-1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def fit_head(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        lr: float = 1e-2,
        batch_size: int = 32,
    ) -> "HybridClassifier":
        """
        在已训练好量子部分的基础上训练经典分类头。

        典型流程:
            1. 先用 PQCClassifier 训练好一个 backbone
            2. 冻结量子部分，只训练经典头
        """
        torch = self.torch
        nn = self.nn

        # 一次性提取所有特征（量子部分冻结）
        feats = self.extract_features(X).astype(np.float32)
        feats_t = torch.from_numpy(feats)
        y_t = torch.from_numpy(y.astype(np.int64))

        opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            idx = torch.randperm(len(feats_t))
            total_loss = 0.0
            for start in range(0, len(feats_t), batch_size):
                bi = idx[start : start + batch_size]
                opt.zero_grad()
                logits = self.head(feats_t[bi])
                loss = criterion(logits, y_t[bi])
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
        return self

    def save(self, path: str) -> None:
        """保存整个混合模型。"""
        torch = self.torch
        state = {
            "quantum_state": {
                "weights": self.quantum.weights,
                "n_data_qubits": self.quantum.n_data_qubits,
                "n_anc_qubits": self.quantum.n_anc_qubits,
                "n_layers": self.quantum.n_layers,
                "n_classes": self.quantum.n_classes,
                "seed": self.quantum.seed,
            },
            "head_state": self.head.state_dict(),
            "n_classes": self.n_classes,
            "feature_dim": self.feature_dim,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, noise: Optional[NoiseChannel] = None) -> "HybridClassifier":
        import torch as _torch
        state = _torch.load(path, weights_only=False)
        q = state["quantum_state"]
        obj = cls(
            n_data_qubits=q["n_data_qubits"],
            n_anc_qubits=q["n_anc_qubits"],
            n_layers=q["n_layers"],
            n_classes=state["n_classes"],
            noise=noise,
            seed=q["seed"],
        )
        obj.quantum.weights = q["weights"]
        obj.quantum.n_classes = q["n_classes"]
        obj.head.load_state_dict(state["head_state"])
        return obj


# ---------------------------------------------------------------------------
# 量子集成分类器（baseline）
# ---------------------------------------------------------------------------

class QuantumEnsemble:
    """多个 PQC 简单平均的集成分类器，主要用作鲁棒性 baseline。"""

    def __init__(self, classifiers: List[PQCClassifier]):
        assert len(classifiers) > 0
        self.classifiers = classifiers

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.stack([c.predict_proba(X) for c in self.classifiers], axis=0)
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
