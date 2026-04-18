"""
攻击评估指标模块。

提供一组标准指标用于评估对抗攻击的有效性与不可感知性：

    attack_success_rate       - 攻击成功率（最常用）
    misclassification_rate    - 直接误分类率
    perturbation_norm         - Lp 范数下的扰动幅度
    fidelity_input            - 输入-扰动的余弦相似度（近似 fidelity）
    transferability           - 跨模型迁移成功率
    robust_accuracy           - 攻击下分类器的鲁棒准确率
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 基础攻击有效性指标
# ---------------------------------------------------------------------------

def attack_success_rate(
    clf,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    攻击成功率 (ASR)。

    只在**原本预测正确**的样本上计算：
        ASR = (# samples where y_adv != y_true and y_clean == y_true)
              / (# samples where y_clean == y_true)

    这是比直接误分类率更有意义的指标——扰动让原本就错的样本继续错不算"成功"。
    """
    y_clean_pred = clf.predict(X_clean)
    correct_mask = y_clean_pred == y_true
    if correct_mask.sum() == 0:
        return 0.0
    y_adv_pred = clf.predict(X_adv[correct_mask])
    return float((y_adv_pred != y_true[correct_mask]).mean())


def misclassification_rate(
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """直接的误分类率（不管原本预测是否正确）。"""
    return float((clf.predict(X) != y).mean())


def robust_accuracy(
    clf,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    鲁棒准确率：在对抗样本下的分类准确率。

    与攻击成功率互补:
        robust_acc ≈ 1 - ASR * clean_acc
    """
    return float((clf.predict(X_adv) == y_true).mean())


# ---------------------------------------------------------------------------
# 扰动不可感知性指标
# ---------------------------------------------------------------------------

def perturbation_norm(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    norm: str = "linf",
) -> np.ndarray:
    """
    按样本计算扰动的 Lp 范数。

    Returns
    -------
    ndarray, shape (N,)
        每个样本的扰动幅度。
    """
    delta = X_adv - X_clean
    if norm == "linf":
        return np.abs(delta).max(axis=1)
    elif norm == "l2":
        return np.linalg.norm(delta, axis=1)
    elif norm == "l1":
        return np.abs(delta).sum(axis=1)
    else:
        raise ValueError(f"unknown norm: {norm}")


def fidelity_input(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    输入-对抗样本的余弦相似度（归一化内积）。

    对振幅编码来说这个量直接对应两个量子态的内积平方（保真度）。
    值接近 1 表示扰动小，越接近 0 表示扰动越大。
    """
    # 按样本归一化
    a = X_clean / (np.linalg.norm(X_clean, axis=1, keepdims=True) + eps)
    b = X_adv / (np.linalg.norm(X_adv, axis=1, keepdims=True) + eps)
    return np.abs((a * b).sum(axis=1)) ** 2


# ---------------------------------------------------------------------------
# 迁移性指标
# ---------------------------------------------------------------------------

def transferability(
    source_clf,
    target_clf,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    对抗样本从 source_clf 迁移到 target_clf 的成功率。

    在 target_clf 原本预测正确的样本上，有多少在 X_adv 下被错误分类。
    """
    mask = target_clf.predict(X_clean) == y_true
    if mask.sum() == 0:
        return 0.0
    pred_adv = target_clf.predict(X_adv[mask])
    return float((pred_adv != y_true[mask]).mean())


# ---------------------------------------------------------------------------
# 组合报告
# ---------------------------------------------------------------------------

def attack_report(
    clf,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    norm: str = "linf",
) -> dict:
    """生成一个攻击结果的完整指标字典。"""
    return {
        "clean_accuracy": float((clf.predict(X_clean) == y_true).mean()),
        "robust_accuracy": robust_accuracy(clf, X_adv, y_true),
        "attack_success_rate": attack_success_rate(clf, X_clean, X_adv, y_true),
        "misclassification_rate": misclassification_rate(clf, X_adv, y_true),
        "perturbation_norm_mean": float(
            perturbation_norm(X_clean, X_adv, norm=norm).mean()
        ),
        "perturbation_norm_max": float(
            perturbation_norm(X_clean, X_adv, norm=norm).max()
        ),
        "fidelity_mean": float(fidelity_input(X_clean, X_adv).mean()),
        "n_samples": int(len(X_clean)),
    }


def print_report(report: dict, title: str = "Attack Report") -> None:
    """友好地打印报告字典。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'=' * 60}\n")
