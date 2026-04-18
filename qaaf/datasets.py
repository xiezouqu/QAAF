"""
数据集加载与预处理模块。

本模块负责把经典数据集转换成 PQC 可以接收的格式：
    1. 下载原始数据（torchvision）
    2. 按需下采样到小分辨率（如 16×16、8×8）
    3. 选择指定的类别子集（如仅取 0、1 两类做二分类）
    4. 展平为向量、归一化
    5. padding 到 2 的整数次方（振幅编码要求 d = 2^n）

提供的接口:
    load_mnist       - 加载 MNIST 并按需下采样 / 选类
    load_fmnist      - 加载 Fashion-MNIST
    load_synthetic   - 生成合成数据（用于快速单元测试）
    amplitude_normalize  - 振幅编码前的归一化

Notes
-----
真实实验中数据加载常常成为 PQC 训练的最大开销来源（因为每个样本都要做电路评估）。
所以这里支持以 "n_per_class" 形式对每类采样固定数量样本，便于快速实验。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MNIST / FMNIST
# ---------------------------------------------------------------------------

def _load_torchvision_dataset(
    name: str,
    root: str,
    train: bool,
):
    """惰性加载 torchvision 数据集，只在真正需要时导入。"""
    from torchvision import datasets, transforms

    tfm = transforms.ToTensor()
    if name.lower() == "mnist":
        ds = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
    elif name.lower() in ("fmnist", "fashion_mnist"):
        ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=tfm)
    else:
        raise ValueError(f"unsupported dataset: {name}")

    X = np.stack([np.array(img).squeeze() for img, _ in ds], axis=0).astype(np.float32)
    y = np.array([label for _, label in ds], dtype=np.int64)
    return X, y


def downsample(X: np.ndarray, size: int) -> np.ndarray:
    """
    把图像下采样到 (size, size)。

    使用简单的 block average pooling 而非双线性插值，保证实现确定性。
    """
    h, w = X.shape[1], X.shape[2]
    assert h % size == 0 and w % size == 0, (
        f"Image size ({h}x{w}) must be divisible by target size {size}"
    )
    bh, bw = h // size, w // size
    X_ds = X.reshape(X.shape[0], size, bh, size, bw).mean(axis=(2, 4))
    return X_ds.astype(np.float32)


def select_classes(
    X: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """选择指定类别，并重新映射标签到 [0, n_classes)。"""
    mask = np.isin(y, classes)
    X_sel = X[mask]
    y_sel = y[mask]
    # 重新映射
    mapping = {c: i for i, c in enumerate(classes)}
    y_new = np.array([mapping[int(v)] for v in y_sel], dtype=np.int64)
    return X_sel, y_new


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """每个类随机采样 n_per_class 个样本，构造类均衡的小数据集。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    X_out, y_out = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        X_out.append(X[chosen])
        y_out.append(y[chosen])
    X_out = np.concatenate(X_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    # 打乱
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def amplitude_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    振幅编码要求每个样本 L2 归一化。

    对 MNIST 这种像素值在 [0, 1] 的图像，若所有像素为 0 归一化会出问题，
    这里用 eps 做平滑保护。
    """
    norms = np.linalg.norm(X.reshape(len(X), -1), axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    X_flat = X.reshape(len(X), -1) / norms
    return X_flat


def pad_to_power_of_two(X: np.ndarray) -> np.ndarray:
    """对向量末尾补零到 2 的整数次方维度。"""
    d = X.shape[1]
    d_pad = 1 << int(np.ceil(np.log2(max(d, 2))))
    if d_pad == d:
        return X
    pad = np.zeros((len(X), d_pad - d), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


# ---------------------------------------------------------------------------
# 顶层加载接口
# ---------------------------------------------------------------------------

def load_mnist(
    size: int = 16,
    classes: Sequence[int] = (0, 1),
    n_per_class: Optional[int] = None,
    train: bool = True,
    data_root: str = "./data",
    normalize_for_quantum: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载并预处理 MNIST。

    Parameters
    ----------
    size : int
        下采样目标尺寸。默认 16×16 = 256 维 = 8 qubit 承载。
    classes : sequence
        选择的类别编号。
    n_per_class : int | None
        每类保留的样本数（None = 全部）。
    train : bool
        True 返回训练集，False 返回测试集。
    data_root : str
        数据存储目录。
    normalize_for_quantum : bool
        是否做振幅归一化 + padding。False 时返回原始展平像素（[0, 1]）。

    Returns
    -------
    X : ndarray, shape (N, d_padded)
    y : ndarray, shape (N,)
    """
    X_raw, y_raw = _load_torchvision_dataset("mnist", data_root, train)
    X_raw = downsample(X_raw, size)
    X_sel, y_sel = select_classes(X_raw, y_raw, classes)

    if n_per_class is not None:
        X_sel, y_sel = balance_classes(X_sel, y_sel, n_per_class)

    if normalize_for_quantum:
        X_flat = amplitude_normalize(X_sel)
        X_flat = pad_to_power_of_two(X_flat)
        return X_flat, y_sel
    else:
        return X_sel.reshape(len(X_sel), -1), y_sel


def load_fmnist(
    size: int = 16,
    classes: Sequence[int] = (0, 1),
    n_per_class: Optional[int] = None,
    train: bool = True,
    data_root: str = "./data",
    normalize_for_quantum: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载并预处理 Fashion-MNIST。接口与 load_mnist 相同。"""
    X_raw, y_raw = _load_torchvision_dataset("fmnist", data_root, train)
    X_raw = downsample(X_raw, size)
    X_sel, y_sel = select_classes(X_raw, y_raw, classes)
    if n_per_class is not None:
        X_sel, y_sel = balance_classes(X_sel, y_sel, n_per_class)
    if normalize_for_quantum:
        X_flat = amplitude_normalize(X_sel)
        X_flat = pad_to_power_of_two(X_flat)
        return X_flat, y_sel
    else:
        return X_sel.reshape(len(X_sel), -1), y_sel


# ---------------------------------------------------------------------------
# 合成数据 (快速测试用)
# ---------------------------------------------------------------------------

def load_synthetic(
    n_samples: int = 200,
    n_features: int = 8,
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合成分类数据，用于 PQC 流程的快速自检。

    构造方式:
        - 每类对应一个随机中心向量（单位球面上）
        - 样本 = 中心 + 小高斯噪声，再归一化

    这个数据集能让一个浅层 PQC 很快达到 ~95% 准确率，
    便于验证训练/攻击流程是否正确。
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    X, y = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        samples = centers[c] + 0.3 * rng.standard_normal((per_class, n_features))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        X.append(samples.astype(np.float32))
        y.extend([c] * per_class)
    X = np.concatenate(X, axis=0)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# 数据分割
# ---------------------------------------------------------------------------

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按比例随机划分训练/测试集。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_test = int(n * test_size)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
