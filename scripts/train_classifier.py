#!/usr/bin/env python
"""
训练量子分类器主脚本。

用法:
    python scripts/train_classifier.py --config configs/mnist_2class.yaml
    python scripts/train_classifier.py --dataset mnist --classes 0,1 --depth 10

训练好的模型权重会保存到 results/classifiers/ 目录下。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# 添加项目根到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qaaf import circuits, classifiers, datasets, utils
from qaaf.noise_models import build_noise


def parse_args():
    p = argparse.ArgumentParser(description="Train a PQC classifier.")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file. If set, other args are overridden.")
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "synthetic"])
    p.add_argument("--classes", type=str, default="0,1",
                   help="Comma-separated class indices, e.g. '0,1'")
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--depth", type=int, default=10,
                   help="PQC variational layer count.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/classifiers")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag appended to the output filename.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = {}
    if args.config is not None:
        cfg = load_config(args.config)

    # 合并 config 与命令行（命令行优先）
    def get(key, default):
        if args.config is not None and key in cfg:
            return cfg[key]
        return default

    dataset_name = get("dataset", args.dataset)
    classes = [int(c) for c in str(get("classes", args.classes)).split(",")]
    image_size = get("image_size", args.image_size)
    n_per_class = get("n_per_class", args.n_per_class)
    n_layers = get("depth", args.depth)
    epochs = get("epochs", args.epochs)
    batch_size = get("batch_size", args.batch_size)
    lr = get("lr", args.lr)
    seed = get("seed", args.seed)
    output_dir = get("output_dir", args.output_dir)
    tag = get("tag", args.tag)
    noise_cfg = cfg.get("noise", None)

    utils.set_seed(seed)
    logger = utils.get_logger(
        "train_classifier",
        log_file=f"logs/train_classifier_{tag or 'default'}.log",
    )

    logger.info(f"Dataset: {dataset_name}, classes: {classes}")
    logger.info(f"PQC: depth={n_layers}, image_size={image_size}")
    logger.info(f"Training: epochs={epochs}, batch={batch_size}, lr={lr}")

    # -----------------------------------------------------------------
    # 加载数据
    # -----------------------------------------------------------------
    if dataset_name == "mnist":
        X_train, y_train = datasets.load_mnist(
            size=image_size, classes=classes,
            n_per_class=n_per_class, train=True,
        )
        X_test, y_test = datasets.load_mnist(
            size=image_size, classes=classes,
            n_per_class=max(50, n_per_class // 4), train=False,
        )
    elif dataset_name == "fmnist":
        X_train, y_train = datasets.load_fmnist(
            size=image_size, classes=classes,
            n_per_class=n_per_class, train=True,
        )
        X_test, y_test = datasets.load_fmnist(
            size=image_size, classes=classes,
            n_per_class=max(50, n_per_class // 4), train=False,
        )
    elif dataset_name == "synthetic":
        X, y = datasets.load_synthetic(
            n_samples=n_per_class * len(classes) * 2,
            n_features=image_size * image_size,
            n_classes=len(classes),
            seed=seed,
        )
        X_train, X_test, y_train, y_test = datasets.train_test_split(X, y, 0.2, seed)
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")

    d = X_train.shape[1]
    n_data_qubits = circuits.n_qubits_for_input_dim(d)
    n_anc_qubits = circuits.n_anc_for_classes(len(classes))
    logger.info(
        f"Input dim={d}, n_data_qubits={n_data_qubits}, "
        f"n_anc_qubits={n_anc_qubits}"
    )
    logger.info(f"Train size={len(X_train)}, test size={len(X_test)}")

    # -----------------------------------------------------------------
    # 构造分类器
    # -----------------------------------------------------------------
    noise = build_noise(noise_cfg) if noise_cfg else None
    clf = classifiers.PQCClassifier(
        n_data_qubits=n_data_qubits,
        n_anc_qubits=n_anc_qubits,
        n_layers=n_layers,
        noise=noise,
        seed=seed,
    )

    # -----------------------------------------------------------------
    # 训练
    # -----------------------------------------------------------------
    train_cfg = classifiers.TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        verbose=True,
    )
    clf.fit(X_train, y_train, train_cfg, X_val=X_test, y_val=y_test)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    logger.info(f"Final train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # -----------------------------------------------------------------
    # 保存
    # -----------------------------------------------------------------
    tag_str = tag or f"{dataset_name}_{len(classes)}c_d{n_layers}"
    out_path = Path(output_dir) / f"{tag_str}.pkl"
    clf.save(str(out_path))
    logger.info(f"Saved classifier to {out_path}")

    # 也保存一个简短的训练信息 JSON
    info = {
        "dataset": dataset_name,
        "classes": classes,
        "image_size": image_size,
        "n_layers": n_layers,
        "n_data_qubits": n_data_qubits,
        "n_anc_qubits": n_anc_qubits,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }
    utils.save_json(info, str(out_path.with_suffix(".info.json")))


if __name__ == "__main__":
    main()
