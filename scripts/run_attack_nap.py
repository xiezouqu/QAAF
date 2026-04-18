#!/usr/bin/env python
"""
运行 NAP 攻击（贡献 2）。

Noise-Aware Perturbation. 在含噪电路下通过蒙特卡洛期望化梯度得到稳定扰动。

用法:
    python scripts/run_attack_nap.py \
        --classifier results/classifiers/mnist_2c_d10.pkl \
        --noise-type depolarizing --noise-strength 0.05 \
        --mc-samples 16 --epsilon 0.1 --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import attacks, classifiers, datasets, metrics, utils
from qaaf.noise_models import build_noise, NoiseConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "synthetic"])
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-test", type=int, default=30,
                   help="NAP 比 PSA 慢很多，测试集默认更小。")
    p.add_argument("--noise-type", type=str, default="depolarizing",
                   choices=["depolarizing", "bit_flip", "phase_flip",
                            "amplitude_damping", "none"])
    p.add_argument("--noise-strength", type=float, default=0.05)
    p.add_argument("--mc-samples", type=int, default=16,
                   help="每步蒙特卡洛采样数。")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--output", type=str,
                   default="results/attacks/nap_result.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("run_nap", log_file="logs/run_attack_nap.log")

    # 构造噪声信道，加载分类器并重新绑定噪声
    noise = build_noise(NoiseConfig(
        type=args.noise_type, strength=args.noise_strength
    ))
    clf = classifiers.PQCClassifier.load(args.classifier, noise=noise)
    clf.noise = noise
    clf._qnode = None  # 触发重建
    logger.info(f"Loaded classifier with noise: {noise}")

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

    if len(X) > args.n_test:
        X, y = X[: args.n_test], y[: args.n_test]
    logger.info(f"Attacking {len(X)} samples with MC={args.mc_samples}")

    # 构造攻击
    config = attacks.NAPConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        alpha=args.alpha,
        norm=args.norm,
        mc_samples=args.mc_samples,
    )
    attacker = attacks.NoiseAwarePerturbation(clf, config)

    X_adv = attacker.generate(X, y)

    # 评估
    report = metrics.attack_report(clf, X, X_adv, y, norm=args.norm)
    report.update({
        "attack_kind": "NAP",
        "noise_type": args.noise_type,
        "noise_strength": args.noise_strength,
        "mc_samples": args.mc_samples,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "dataset": args.dataset,
    })

    metrics.print_report(report, title="NAP attack result")
    utils.save_json(report, args.output)
    logger.info(f"Report saved to {args.output}")

    npz_path = Path(args.output).with_suffix(".npz")
    np.savez(npz_path, X=X, y=y, X_adv=X_adv)
    logger.info(f"Adversarial samples saved to {npz_path}")


if __name__ == "__main__":
    main()
