#!/usr/bin/env python
"""
实验：噪声强度扫描。

固定分类器与攻击预算 epsilon，扫描去极化噪声强度从 0 到 0.1，
对比 PSA vs NAP 在不同噪声下的攻击成功率。

这是对贡献 2 (NAP) 的核心验证实验。

用法:
    python experiments/noise_sweep.py \
        --classifier results/classifiers/mnist_2c_d10.pkl \
        --noise-values 0 0.01 0.03 0.05 0.08 0.1
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
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-test", type=int, default=20,
                   help="每个噪声点用的测试样本数")
    p.add_argument("--noise-type", type=str, default="depolarizing")
    p.add_argument("--noise-values", type=float, nargs="+",
                   default=[0.0, 0.01, 0.03, 0.05, 0.08, 0.1])
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--output", type=str,
                   default="results/experiments/noise_sweep.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_single(clf, attacker, X, y, norm):
    X_adv = attacker.generate(X, y)
    return metrics.attack_report(clf, X, X_adv, y, norm=norm)


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("noise_sweep", log_file="logs/noise_sweep.log")

    # 数据
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
            n_samples=args.n_test * len(classes) * 2,
            n_features=args.image_size * args.image_size,
            n_classes=len(classes), seed=args.seed,
        )

    if len(X) > args.n_test:
        X, y = X[: args.n_test], y[: args.n_test]
    logger.info(f"Sweeping {len(args.noise_values)} noise values on {len(X)} samples")

    # -----------------------------------------------------------------
    # 扫描
    # -----------------------------------------------------------------
    results = {"noise_values": list(args.noise_values), "psa": [], "nap": []}

    for s in args.noise_values:
        logger.info(f"=== noise strength = {s} ===")
        # 每次构造新的 classifier 以绑定不同噪声
        noise = build_noise(NoiseConfig(type=args.noise_type, strength=s))
        clf = classifiers.PQCClassifier.load(args.classifier, noise=noise)
        clf.noise = noise
        clf._qnode = None

        # 1) PSA (无噪声感知)
        psa_cfg = attacks.PSAConfig(
            epsilon=args.epsilon, steps=args.steps,
            alpha=args.alpha, variant="pgd", verbose=False,
        )
        psa = attacks.ParameterShiftAttack(clf, psa_cfg)
        r_psa = run_single(clf, psa, X, y, norm="linf")
        r_psa["noise_strength"] = s
        logger.info(f"PSA @ noise={s}: ASR={r_psa['attack_success_rate']:.4f}")
        results["psa"].append(r_psa)

        # 2) NAP (噪声感知)
        nap_cfg = attacks.NAPConfig(
            epsilon=args.epsilon, steps=args.steps,
            alpha=args.alpha, mc_samples=args.mc_samples, verbose=False,
        )
        nap = attacks.NoiseAwarePerturbation(clf, nap_cfg)
        r_nap = run_single(clf, nap, X, y, norm="linf")
        r_nap["noise_strength"] = s
        logger.info(f"NAP @ noise={s}: ASR={r_nap['attack_success_rate']:.4f}")
        results["nap"].append(r_nap)

    utils.save_json(results, args.output)
    logger.info(f"Full results saved to {args.output}")

    # 打印对比表
    print("\n=== Noise sweep summary ===")
    print(f"{'noise':>8s}  {'PSA_ASR':>10s}  {'NAP_ASR':>10s}")
    for s, rp, rn in zip(
        args.noise_values, results["psa"], results["nap"]
    ):
        print(
            f"{s:>8.3f}  {rp['attack_success_rate']:>10.4f}  "
            f"{rn['attack_success_rate']:>10.4f}"
        )


if __name__ == "__main__":
    main()
