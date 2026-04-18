#!/usr/bin/env python
"""
结果可视化脚本。

从 results/attacks/ 下读取 JSON 报告，绘制:
    - 扰动强度 vs 攻击成功率曲线
    - 噪声强度 vs 攻击成功率曲线 (NAP)
    - CMJA 三种模式的柱状对比

用法:
    python scripts/plot_results.py --kind psa --pattern "results/attacks/psa_eps*.json"
    python scripts/plot_results.py --kind nap --pattern "results/attacks/nap_noise*.json"
    python scripts/plot_results.py --kind cmja --input results/attacks/cmja_result.json
    python scripts/plot_results.py --kind samples --npz results/attacks/psa_result.npz

生成的图存到 results/figures/ 目录下。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, required=True,
                   choices=["psa", "nap", "cmja", "samples"])
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob pattern for multiple JSON reports (psa/nap).")
    p.add_argument("--input", type=str, default=None,
                   help="Single JSON/NPZ input (cmja/samples).")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def plot_psa_curve(pattern: str, output: str):
    """绘制 PSA 下不同 epsilon 对应的 ASR 曲线。"""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match {pattern}")
        return
    eps_list, asr_list = [], []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        eps_list.append(r["epsilon"])
        asr_list.append(r["attack_success_rate"])

    order = np.argsort(eps_list)
    eps_list = np.array(eps_list)[order]
    asr_list = np.array(asr_list)[order]

    plt.figure(figsize=(5, 4))
    plt.plot(eps_list, asr_list, "o-", linewidth=2, markersize=6)
    plt.xlabel(r"Perturbation Strength $\epsilon$")
    plt.ylabel("Attack Success Rate")
    plt.title("PSA: ASR vs perturbation strength")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_nap_curve(pattern: str, output: str):
    """绘制 NAP 下不同噪声强度对应的 ASR 曲线。"""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match {pattern}")
        return
    noise_list, asr_list, mc_list = [], [], []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        noise_list.append(r["noise_strength"])
        asr_list.append(r["attack_success_rate"])
        mc_list.append(r.get("mc_samples", 1))

    order = np.argsort(noise_list)
    noise_list = np.array(noise_list)[order]
    asr_list = np.array(asr_list)[order]

    plt.figure(figsize=(5, 4))
    plt.plot(noise_list, asr_list, "s-", linewidth=2,
             markersize=6, color="tab:red")
    plt.xlabel("Depolarizing Noise Strength")
    plt.ylabel("Attack Success Rate")
    plt.title("NAP: ASR vs noise strength")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_cmja_bar(input_path: str, output: str):
    """绘制 CMJA 三种模式的 ASR 柱状对比。"""
    with open(input_path) as fp:
        r = json.load(fp)
    modes = ["joint", "quantum_only", "classical_only"]
    asrs = [r["reports"][m]["attack_success_rate"] for m in modes]
    robusts = [r["reports"][m]["robust_accuracy"] for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, asrs, width, label="ASR", color="tab:red")
    ax.bar(x + width / 2, robusts, width, label="Robust Acc", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(["Joint", "Quantum\nOnly", "Classical\nOnly"])
    ax.set_ylabel("Rate")
    ax.set_title("CMJA mode comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_samples(npz_path: str, output: str, max_show: int = 8):
    """可视化几个原始样本 + 对抗样本的对比。"""
    d = np.load(npz_path)
    X, X_adv, y = d["X"], d["X_adv"], d["y"]
    n = min(max_show, len(X))
    # 假设图像是 16×16 = 256 维 (padding 到 256 或 512)
    d_img = int(np.sqrt(X.shape[1])) if X.shape[1] in (256, 64, 16) else None
    if d_img is None:
        # 尝试最接近的平方数
        d_img = int(np.floor(np.sqrt(X.shape[1])))

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        clean_img = X[i, : d_img * d_img].reshape(d_img, d_img)
        adv_img = X_adv[i, : d_img * d_img].reshape(d_img, d_img)
        axes[0, i].imshow(clean_img, cmap="gray")
        axes[0, i].set_title(f"y={y[i]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(adv_img, cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel("Adv", fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def main():
    args = parse_args()
    output = args.output or str(FIG_DIR / f"{args.kind}.png")

    if args.kind == "psa":
        plot_psa_curve(args.pattern, output)
    elif args.kind == "nap":
        plot_nap_curve(args.pattern, output)
    elif args.kind == "cmja":
        plot_cmja_bar(args.input, output)
    elif args.kind == "samples":
        plot_samples(args.input, output)
    else:
        raise ValueError(f"unknown kind: {args.kind}")


if __name__ == "__main__":
    main()
