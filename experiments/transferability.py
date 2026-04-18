#!/usr/bin/env python
"""
实验：攻击迁移性测试。

在一个 source 分类器上生成对抗样本，测试它们在:
    1. 同架构不同种子的 target 分类器上的成功率
    2. 不同架构 (不同深度) 的 target 分类器上的成功率
    3. 经典 CNN 上的成功率 (如果提供了经典基线)

这是一个标准的黑盒攻击评估实验。

用法:
    python experiments/transferability.py \
        --source results/classifiers/mnist_2c_d10_seed42.pkl \
        --targets results/classifiers/mnist_2c_d10_seed0.pkl \
                  results/classifiers/mnist_2c_d8.pkl \
                  results/classifiers/mnist_2c_d16.pkl
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
    p.add_argument("--source", type=str, required=True,
                   help="生成对抗样本的 source 分类器。")
    p.add_argument("--targets", type=str, nargs="+", required=True,
                   help="要迁移测试的 target 分类器列表。")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--classes", type=str, default="0,1")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--variant", type=str, default="mifgsm",
                   choices=["fgsm", "pgd", "mifgsm"],
                   help="默认 MIFGSM 因为动量能提升迁移性。")
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--output", type=str,
                   default="results/experiments/transferability.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("transferability",
                              log_file="logs/transferability.log")

    # 加载 source
    src_clf = classifiers.PQCClassifier.load(args.source)
    logger.info(f"Source: {args.source}")
    logger.info(
        f"  n_layers={src_clf.n_layers}, n_qubits={src_clf.n_qubits_total}"
    )

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
            n_samples=args.n_test * len(classes),
            n_features=args.image_size * args.image_size,
            n_classes=len(classes), seed=args.seed,
        )
    if len(X) > args.n_test:
        X, y = X[: args.n_test], y[: args.n_test]
    logger.info(f"Test set size: {len(X)}")

    # 生成对抗样本
    logger.info(f"Generating adversarial samples with {args.variant}...")
    cfg = attacks.PSAConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        alpha=args.alpha,
        variant=args.variant,
        verbose=False,
    )
    atk = attacks.ParameterShiftAttack(src_clf, cfg)
    X_adv = atk.generate(X, y)

    # 先评估 source 自己上的效果（白盒 ASR 作参考）
    src_report = metrics.attack_report(src_clf, X, X_adv, y, norm="linf")
    logger.info(
        f"White-box on source: ASR={src_report['attack_success_rate']:.4f}"
    )

    # 逐个评估 targets
    transfer_results = []
    for tgt_path in args.targets:
        tgt_clf = classifiers.PQCClassifier.load(tgt_path)
        clean_acc = tgt_clf.score(X, y)
        tr_rate = metrics.transferability(src_clf, tgt_clf, X, X_adv, y)
        robust_acc = metrics.robust_accuracy(tgt_clf, X_adv, y)
        logger.info(
            f"{Path(tgt_path).name}: clean_acc={clean_acc:.4f}, "
            f"transfer_ASR={tr_rate:.4f}, robust_acc={robust_acc:.4f}"
        )
        transfer_results.append({
            "target": str(tgt_path),
            "n_layers": tgt_clf.n_layers,
            "clean_accuracy": clean_acc,
            "transfer_asr": tr_rate,
            "robust_accuracy": robust_acc,
        })

    summary = {
        "source": args.source,
        "variant": args.variant,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "white_box_asr": src_report["attack_success_rate"],
        "targets": transfer_results,
    }
    utils.save_json(summary, args.output)
    logger.info(f"Saved summary to {args.output}")

    # 打印对比
    print("\n=== Transferability summary ===")
    print(f"White-box (source) ASR: {src_report['attack_success_rate']:.4f}\n")
    print(f"{'target':<50s} {'ASR':>8s} {'robust':>10s}")
    for r in transfer_results:
        print(f"{Path(r['target']).name:<50s} "
              f"{r['transfer_asr']:>8.4f} "
              f"{r['robust_accuracy']:>10.4f}")


if __name__ == "__main__":
    main()
