#!/usr/bin/env python
"""
单元测试：验证所有核心模块能正常工作。

快速自检脚本，不依赖真实数据集（用 synthetic）。在修改代码后先跑这个
确认没破坏基本链路：

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import (
    attacks,
    circuits,
    classifiers,
    datasets,
    gradient,
    metrics,
    noise_models,
    utils,
)


def test_utils():
    utils.set_seed(123)
    a = utils.normalize_vector(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(a), 1.0)

    delta = np.array([0.5, -0.8, 0.2])
    clipped = utils.clip_perturbation(delta, epsilon=0.3, norm="linf")
    assert np.all(np.abs(clipped) <= 0.3 + 1e-9)
    print("  [ok] utils")


def test_noise_models():
    for cfg in [
        {"type": "none"},
        {"type": "depolarizing", "strength": 0.05},
        {"type": "bit_flip", "strength": 0.02},
        {"type": "phase_flip", "strength": 0.01},
        {"type": "amplitude_damping", "strength": 0.03},
    ]:
        ch = noise_models.build_noise(cfg)
        ks = ch.kraus_operators()
        assert len(ks) >= 1
    print("  [ok] noise_models")


def test_datasets_synthetic():
    X, y = datasets.load_synthetic(
        n_samples=80, n_features=16, n_classes=2, seed=1
    )
    assert X.shape[0] == len(y) == 80
    # 振幅归一化
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    print("  [ok] datasets.load_synthetic")


def test_circuits():
    n_data = circuits.n_qubits_for_input_dim(8)
    assert n_data == 3
    n_anc = circuits.n_anc_for_classes(2)
    assert n_anc == 1
    w = circuits.init_classifier_weights(n_layers=3, n_qubits=4, seed=0)
    assert w.shape == (3, 4, 3)
    print("  [ok] circuits helpers")


def test_classifier_fit_predict():
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=60, n_features=8, n_classes=2, seed=42
    )
    n_data = circuits.n_qubits_for_input_dim(X.shape[1])
    clf = classifiers.PQCClassifier(
        n_data_qubits=n_data, n_anc_qubits=1, n_layers=3, seed=42,
    )
    cfg = classifiers.TrainConfig(epochs=5, batch_size=16, lr=0.05, verbose=False)
    clf.fit(X, y, cfg)
    acc = clf.score(X, y)
    # 合成数据+3层+5个epoch 应该至少显著优于随机
    assert acc > 0.55, f"expected > 0.55, got {acc}"
    # 序列化
    save_path = "/tmp/qaaf_test_clf.pkl"
    clf.save(save_path)
    clf2 = classifiers.PQCClassifier.load(save_path)
    assert np.allclose(clf2.weights, clf.weights)
    print(f"  [ok] classifier fit/predict (acc={acc:.3f})")


def test_gradient_consistency():
    """验证 analytic 和 finite-difference 梯度方向一致。"""
    utils.set_seed(0)
    X, y = datasets.load_synthetic(
        n_samples=20, n_features=8, n_classes=2, seed=0
    )
    clf = classifiers.PQCClassifier(
        n_data_qubits=3, n_anc_qubits=1, n_layers=2, seed=0
    )
    clf.n_classes = 2  # 不训练, 直接打标签数

    x, label = X[0], int(y[0])
    g_ana = gradient.analytic_input_gradient(clf, x, label)
    g_fd = gradient.finite_difference_gradient(clf, x, label, eps=5e-3)
    # 相关系数应该很接近 1
    cos = np.dot(g_ana, g_fd) / (
        np.linalg.norm(g_ana) * np.linalg.norm(g_fd) + 1e-12
    )
    assert cos > 0.5, f"analytic vs fd cosine too low: {cos}"
    print(f"  [ok] gradient consistency (cosine={cos:.3f})")


def test_psa_attack():
    """PSA 单步攻击应该改变预测概率。"""
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=30, n_features=8, n_classes=2, seed=42
    )
    clf = classifiers.PQCClassifier(
        n_data_qubits=3, n_anc_qubits=1, n_layers=3, seed=42,
    )
    clf.fit(X, y, classifiers.TrainConfig(epochs=4, batch_size=16, verbose=False))

    cfg = attacks.PSAConfig(
        epsilon=0.2, steps=10, alpha=0.05, variant="pgd", verbose=False,
    )
    atk = attacks.ParameterShiftAttack(clf, cfg)
    X_adv = atk.generate(X[:5], y[:5])
    # 扰动幅度小于等于 epsilon
    pert = metrics.perturbation_norm(X[:5], X_adv, norm="linf")
    assert pert.max() <= 0.2 + 1e-5, f"perturbation exceeds eps: {pert.max()}"
    print("  [ok] psa attack runs, perturbation within eps")


def test_nap_attack():
    """NAP 在无噪情况下应该接近 PSA。"""
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=20, n_features=8, n_classes=2, seed=42
    )
    clf = classifiers.PQCClassifier(
        n_data_qubits=3, n_anc_qubits=1, n_layers=2, seed=42,
    )
    clf.fit(X, y, classifiers.TrainConfig(epochs=3, batch_size=16, verbose=False))

    cfg = attacks.NAPConfig(
        epsilon=0.2, steps=5, alpha=0.05, mc_samples=2, verbose=False,
    )
    atk = attacks.NoiseAwarePerturbation(clf, cfg)
    X_adv = atk.generate(X[:3], y[:3])
    assert X_adv.shape == X[:3].shape
    print("  [ok] nap attack runs")


def test_hybrid_model():
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=40, n_features=16, n_classes=2, seed=42
    )
    model = classifiers.HybridClassifier(
        n_data_qubits=4, n_anc_qubits=2, n_layers=2,
        n_classes=2, head_hidden=8, seed=42,
    )
    model.quantum.fit(
        X, y,
        classifiers.TrainConfig(epochs=3, batch_size=16, verbose=False),
    )
    model.fit_head(X, y, epochs=10)
    acc = model.score(X, y)
    print(f"  [ok] hybrid model (acc={acc:.3f})")


def test_cmja_attack():
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=20, n_features=16, n_classes=2, seed=42
    )
    model = classifiers.HybridClassifier(
        n_data_qubits=4, n_anc_qubits=2, n_layers=2,
        n_classes=2, head_hidden=8, seed=42,
    )
    model.quantum.fit(
        X, y,
        classifiers.TrainConfig(epochs=2, batch_size=16, verbose=False),
    )
    model.fit_head(X, y, epochs=5)

    cfg = attacks.CMJAConfig(
        epsilon_x=0.1, steps=3, alpha_x=0.03, lam=0.5,
        mode="joint", verbose=False,
    )
    atk = attacks.CrossModuleJointAttack(model, cfg)
    X_adv = atk.generate(X[:3], y[:3])
    assert X_adv.shape == X[:3].shape
    print("  [ok] cmja attack runs")


def test_metrics():
    utils.set_seed(42)
    X, y = datasets.load_synthetic(
        n_samples=20, n_features=8, n_classes=2, seed=42
    )
    clf = classifiers.PQCClassifier(
        n_data_qubits=3, n_anc_qubits=1, n_layers=2, seed=42,
    )
    clf.fit(X, y, classifiers.TrainConfig(epochs=2, batch_size=16, verbose=False))

    X_adv = X + 0.05 * np.random.randn(*X.shape)
    X_adv = X_adv / np.linalg.norm(X_adv, axis=1, keepdims=True)

    report = metrics.attack_report(clf, X, X_adv, y)
    for key in ("clean_accuracy", "robust_accuracy", "attack_success_rate"):
        assert key in report
    print("  [ok] metrics report generated")


ALL_TESTS = [
    test_utils,
    test_noise_models,
    test_datasets_synthetic,
    test_circuits,
    test_classifier_fit_predict,
    test_gradient_consistency,
    test_psa_attack,
    test_nap_attack,
    test_hybrid_model,
    test_cmja_attack,
    test_metrics,
]


def main():
    print("Running QAAF pipeline tests...\n")
    failed = 0
    for t in ALL_TESTS:
        print(f"Test: {t.__name__}")
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {e}")
            traceback.print_exc()
    print(f"\n{'PASSED' if failed == 0 else 'FAILED'}: "
          f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests.")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
