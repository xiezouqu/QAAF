#!/usr/bin/env python
"""
运行 CMJA 攻击（贡献 3）。

Cross-Module Joint Attack. 针对量子-经典混合分类器的联合攻击。

该脚本：
    1. 训练（或加载）一个量子特征提取器
    2. 训练经典分类头，形成混合模型
    3. 在混合模型上分别运行 joint / quantum_only / classical_only 三种攻击
    4. 对比三种攻击的成功率，展示联合攻击的优势

用法:
    python scripts/run_attack_cmja.py \
        --dataset mnist --classes 0,1,2,3 \
        --epsilon 0.1 --steps 40 --lam 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import attacks, circuits, classifiers, datasets, metrics, utils
from qaaf.noise_models import build_noise, NoiseConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", type=str, default=None,
                   help="如果提供,则加载预训练的 HybridClassifier；"
                        "否则脚本会即时训练一个。")
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "synthetic"])
    p.add_argument("--classes", type=str, default="0,1,2,3")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-per-class-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--depth", type=int, default=6,
                   help="混合模型中 PQC 的层数，通常比纯 PQC 少（经典头承担部分能力）")
    p.add_argument("--head-hidden", type=int, default=16)
    p.add_argument("--pqc-epochs", type=int, default=15)
    p.add_argument("--head-epochs", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--lam", type=float, default=0.5,
                   help="量子方向 vs 经典方向的权衡系数")
    p.add_argument("--noise-type", type=str, default="none")
    p.add_argument("--noise-strength", type=float, default=0.0)
    p.add_argument("--output", type=str,
                   default="results/attacks/cmja_result.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args, classes):
    if args.dataset == "mnist":
        X_tr, y_tr = datasets.load_mnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_per_class_train, train=True,
        )
        X_te, y_te = datasets.load_mnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    elif args.dataset == "fmnist":
        X_tr, y_tr = datasets.load_fmnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_per_class_train, train=True,
        )
        X_te, y_te = datasets.load_fmnist(
            size=args.image_size, classes=classes,
            n_per_class=args.n_test, train=False,
        )
    else:
        X, y = datasets.load_synthetic(
            n_samples=args.n_per_class_train * len(classes) * 2,
            n_features=args.image_size * args.image_size,
            n_classes=len(classes),
            seed=args.seed,
        )
        X_tr, X_te, y_tr, y_te = datasets.train_test_split(X, y, 0.2, args.seed)
    return X_tr, y_tr, X_te, y_te


def build_or_load_hybrid(args, X_tr, y_tr, X_te, y_te):
    logger = utils.get_logger("cmja-build")

    if args.classifier is not None:
        noise = build_noise(NoiseConfig(
            type=args.noise_type, strength=args.noise_strength
        )) if args.noise_type != "none" else None
        model = classifiers.HybridClassifier.load(args.classifier, noise=noise)
        logger.info(f"Loaded hybrid from {args.classifier}")
        return model

    # 即时训练
    classes = np.unique(y_tr)
    d = X_tr.shape[1]
    n_data = circuits.n_qubits_for_input_dim(d)
    n_anc = max(2, circuits.n_anc_for_classes(len(classes)))

    noise = build_noise(NoiseConfig(
        type=args.noise_type, strength=args.noise_strength
    )) if args.noise_type != "none" else None

    model = classifiers.HybridClassifier(
        n_data_qubits=n_data,
        n_anc_qubits=n_anc,
        n_layers=args.depth,
        n_classes=len(classes),
        head_hidden=args.head_hidden,
        noise=noise,
        seed=args.seed,
    )
    logger.info("Training PQC backbone...")
    model.quantum.fit(
        X_tr, y_tr,
        classifiers.TrainConfig(
            epochs=args.pqc_epochs, batch_size=32, lr=0.01, seed=args.seed
        ),
    )
    logger.info(f"PQC backbone trained: train_acc={model.quantum.score(X_tr, y_tr):.4f}")

    logger.info("Training classical head...")
    model.fit_head(X_tr, y_tr, epochs=args.head_epochs)
    logger.info(f"Hybrid test_acc={model.score(X_te, y_te):.4f}")

    # 保存
    tag = f"hybrid_{args.dataset}_{len(classes)}c_d{args.depth}"
    save_path = Path("results/classifiers") / f"{tag}.pt"
    model.save(str(save_path))
    logger.info(f"Saved hybrid to {save_path}")
    return model


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    logger = utils.get_logger("cmja", log_file="logs/run_attack_cmja.log")

    classes = [int(c) for c in args.classes.split(",")]
    X_tr, y_tr, X_te, y_te = load_data(args, classes)
    logger.info(f"train_size={len(X_tr)}, test_size={len(X_te)}")

    model = build_or_load_hybrid(args, X_tr, y_tr, X_te, y_te)

    # 限制测试样本数
    if len(X_te) > args.n_test:
        X_te, y_te = X_te[: args.n_test], y_te[: args.n_test]

    # 分别跑三种模式
    all_reports = {}
    for mode in ["joint", "quantum_only", "classical_only"]:
        logger.info(f"Running CMJA in mode={mode}")
        cfg = attacks.CMJAConfig(
            epsilon_x=args.epsilon,
            steps=args.steps,
            alpha_x=args.alpha,
            norm=args.norm,
            lam=args.lam,
            mode=mode,
        )
        atk = attacks.CrossModuleJointAttack(model, cfg)
        X_adv = atk.generate(X_te, y_te)
        rep = metrics.attack_report(model, X_te, X_adv, y_te, norm=args.norm)
        rep["mode"] = mode
        metrics.print_report(rep, title=f"CMJA-{mode}")
        all_reports[mode] = rep

        # 存对抗样本
        npz = Path(args.output).with_name(f"cmja_{mode}.npz")
        np.savez(npz, X=X_te, y=y_te, X_adv=X_adv)

    # 综合保存
    summary = {
        "attack_kind": "CMJA",
        "dataset": args.dataset,
        "classes": classes,
        "epsilon": args.epsilon,
        "steps": args.steps,
        "lam": args.lam,
        "reports": all_reports,
    }
    utils.save_json(summary, args.output)
    logger.info(f"Summary saved to {args.output}")

    # 打印对比
    print("\n=== CMJA mode comparison ===")
    print(f"{'mode':20s} {'ASR':>8s} {'robust_acc':>12s}")
    for mode in ["joint", "quantum_only", "classical_only"]:
        r = all_reports[mode]
        print(f"{mode:20s} {r['attack_success_rate']:>8.4f} "
              f"{r['robust_accuracy']:>12.4f}")


if __name__ == "__main__":
    main()
