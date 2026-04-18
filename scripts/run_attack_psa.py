#!/usr/bin/env python
"""
运行 PSA 攻击（贡献 1）。

Parameter-Shift Adversarial attack. 支持 FGSM / PGD / MIFGSM 三种变体。

用法:
    python scripts/run_attack_psa.py \
        --classifier results/classifiers/mnist_2c_d10.pkl \
        --dataset mnist --classes 0,1 \
        --variant pgd --epsilon 0.1 --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import attacks, classifiers, datasets, metrics, utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "synthetic"])
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-test", type=int, default=50,
                   help="Number of test samples to attack.")
    p.add_argument("--variant", type=str, default="pgd",
                   choices=["fgsm", "pgd", "mifgsm"])
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--grad-method", type=str, default="analytic",
                   choices=["analytic", "fd"])
    p.add_argument("--output", type=str, default="results/attacks/psa_result.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("run_psa", log_file="logs/run_attack_psa.log")

    # 加载分类器
    clf = classifiers.PQCClassifier.load(args.classifier)
    logger.info(f"Loaded classifier from {args.classifier}")
    logger.info(
        f"Classifier: n_data_qubits={clf.n_data_qubits}, "
        f"n_layers={clf.n_layers}, n_classes={clf.n_classes}"
    )

    # 加载测试集
    classes = [int(c) for c in args.classes.split(",")]
    if args.dataset == "mnist":
        X, y = datasets.load_mnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    elif args.dataset == "fmnist":
        X, y = datasets.load_fmnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    else:
        X, y = datasets.load_synthetic(
            n_samples=args.n_test * len(classes),
            n_features=args.image_size * args.image_size,
            n_classes=len(classes),
            seed=args.seed,
        )

    # 只保留 n_test 个样本
    if len(X) > args.n_test:
        idx = np.arange(args.n_test)
        X, y = X[idx], y[idx]

    logger.info(f"Attacking {len(X)} samples")

    # 构造攻击
    config = attacks.PSAConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        alpha=args.alpha,
        norm=args.norm,
        variant=args.variant,
        grad_method=args.grad_method,
    )
    attacker = attacks.ParameterShiftAttack(clf, config)

    # 生成对抗样本
    X_adv = attacker.generate(X, y)

    # 评估
    report = metrics.attack_report(clf, X, X_adv, y, norm=args.norm)
    report["attack_kind"] = "PSA"
    report["variant"] = args.variant
    report["epsilon"] = args.epsilon
    report["steps"] = args.steps
    report["dataset"] = args.dataset
    report["classes"] = classes

    metrics.print_report(report, title=f"PSA-{args.variant} attack result")

    utils.save_json(report, args.output)
    logger.info(f"Report saved to {args.output}")

    # 对抗样本也保存下来便于后续分析
    npz_path = Path(args.output).with_suffix(".npz")
    np.savez(npz_path, X=X, y=y, X_adv=X_adv)
    logger.info(f"Adversarial samples saved to {npz_path}")


if __name__ == "__main__":
    main()
