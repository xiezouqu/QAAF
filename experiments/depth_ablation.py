#!/usr/bin/env python
"""
实验：PQC 电路深度消融。

训练不同深度 (L=4, 8, 12, 16, 20) 的 PQC 分类器，对比:
    1. 干净准确率
    2. PSA 攻击下的鲁棒准确率
    3. 攻击成功率

主要想回答的问题: 更深的 PQC 是否更脆弱(类似经典神经网络越深越容易被攻击)?
还是因为参数更多而反而"含糊"使攻击变难?

用法:
    python experiments/depth_ablation.py \
        --depths 4 8 12 16 --epsilon 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import attacks, circuits, classifiers, datasets, metrics, utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12, 16])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-per-class", type=int, default=150)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--output", type=str,
                   default="results/experiments/depth_ablation.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args, classes):
    if args.dataset == "mnist":
        X_tr, y_tr = datasets.load_mnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_per_class, train=True,
        )
        X_te, y_te = datasets.load_mnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    elif args.dataset == "fmnist":
        X_tr, y_tr = datasets.load_fmnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_per_class, train=True,
        )
        X_te, y_te = datasets.load_fmnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    else:
        X, y = datasets.load_synthetic(
            n_samples=args.n_per_class * len(classes) * 2,
            n_features=args.image_size * args.image_size,
            n_classes=len(classes), seed=args.seed,
        )
        X_tr, X_te, y_tr, y_te = datasets.train_test_split(X, y, 0.2, args.seed)
    return X_tr, y_tr, X_te, y_te


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("depth_ablation",
                              log_file="logs/depth_ablation.log")

    classes = [int(c) for c in args.classes.split(",")]
    X_tr, y_tr, X_te, y_te = load_data(args, classes)
    d = X_tr.shape[1]
    n_data = circuits.n_qubits_for_input_dim(d)
    n_anc = circuits.n_anc_for_classes(len(classes))
    logger.info(f"Input dim={d}, n_data_qubits={n_data}, n_anc_qubits={n_anc}")

    # 限制测试样本
    if len(X_te) > args.n_test:
        X_te, y_te = X_te[: args.n_test], y_te[: args.n_test]

    all_results = {"depths": list(args.depths), "entries": []}

    for depth in args.depths:
        logger.info(f"=== depth = {depth} ===")
        clf = classifiers.PQCClassifier(
            n_data_qubits=n_data,
            n_anc_qubits=n_anc,
            n_layers=depth,
            seed=args.seed,
        )
        clf.fit(
            X_tr, y_tr,
            classifiers.TrainConfig(
                epochs=args.epochs, batch_size=32, lr=0.01,
                seed=args.seed, verbose=False,
            ),
        )
        clean_acc = clf.score(X_te, y_te)
        logger.info(f"  clean test acc = {clean_acc:.4f}")

        # PSA 攻击
        psa_cfg = attacks.PSAConfig(
            epsilon=args.epsilon, steps=args.steps,
            alpha=args.alpha, variant="pgd", verbose=False,
        )
        atk = attacks.ParameterShiftAttack(clf, psa_cfg)
        X_adv = atk.generate(X_te, y_te)
        report = metrics.attack_report(clf, X_te, X_adv, y_te, norm="linf")
        report["depth"] = depth
        report["n_parameters"] = circuits.count_parameters(depth, n_data + n_anc)
        logger.info(
            f"  ASR = {report['attack_success_rate']:.4f}, "
            f"robust_acc = {report['robust_accuracy']:.4f}"
        )
        all_results["entries"].append(report)

    utils.save_json(all_results, args.output)
    logger.info(f"Results saved to {args.output}")

    # 打印对比
    print("\n=== Depth ablation summary ===")
    print(f"{'depth':>6s} {'params':>8s} {'clean_acc':>10s} "
          f"{'ASR':>8s} {'robust_acc':>12s}")
    for r in all_results["entries"]:
        print(f"{r['depth']:>6d} {r['n_parameters']:>8d} "
              f"{r['clean_accuracy']:>10.4f} "
              f"{r['attack_success_rate']:>8.4f} "
              f"{r['robust_accuracy']:>12.4f}")


if __name__ == "__main__":
    main()
