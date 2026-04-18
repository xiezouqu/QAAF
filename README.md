# QAAF: Quantum Adversarial Attack Framework

一个面向参数化量子电路（PQC）分类器的对抗攻击研究原型，聚焦 NISQ 环境下量子机器
学习模型的脆弱性评估。项目目前处于**研究原型阶段**，用于对以下三类核心能力进行
系统性的实验验证：

1. **参数移位规则驱动的量子梯度对抗攻击（Parameter-Shift Adversarial Attack, PSA）**
   —— 针对量子电路不能直接反向传播的问题，使用参数移位规则精确估计量子分类器
   对输入数据的梯度，构建面向纯量子分类器的白盒/黑盒对抗攻击流程。

2. **噪声感知扰动优化（Noise-Aware Perturbation Optimization, NAP）**
   —— 将 NISQ 硬件上常见的去极化噪声、比特翻转噪声显式纳入扰动优化过程，在对抗
   样本生成阶段就对噪声信道进行期望化处理，使扰动在含噪硬件上保持稳定攻击效果。

3. **量子-经典混合模型跨模块联合攻击（Cross-Module Joint Attack, CMJA）**
   —— 针对量子编码器 + 经典分类头组成的混合模型，在量子特征空间与经典决策边界
   上协同施加扰动，对比单模块攻击显著提升了攻击成功率。

## 项目结构

```
QAAF/
├── qaaf/                    # 核心代码库
│   ├── __init__.py
│   ├── circuits.py          # PQC 分类器结构、振幅编码、含噪电路
│   ├── classifiers.py       # 量子分类器、量子-经典混合分类器
│   ├── gradient.py          # 参数移位规则梯度估计
│   ├── attacks.py           # 三类攻击实现 (PSA / NAP / CMJA)
│   ├── noise_models.py      # 去极化/比特翻转噪声信道
│   ├── datasets.py          # MNIST/FMNIST 下采样与振幅编码预处理
│   ├── metrics.py           # 攻击成功率、扰动幅度、迁移率等指标
│   └── utils.py             # 通用工具（随机种子、日志、序列化）
├── scripts/                 # 实验脚本（供论文/答辩复现）
│   ├── train_classifier.py  # 训练量子分类器
│   ├── run_attack_psa.py    # 运行 PSA 攻击实验
│   ├── run_attack_nap.py    # 运行 NAP 噪声感知攻击实验
│   ├── run_attack_cmja.py   # 运行 CMJA 跨模块联合攻击实验
│   └── plot_results.py      # 结果可视化
├── configs/                 # 实验配置 YAML
│   ├── mnist_2class.yaml
│   ├── mnist_4class.yaml
│   └── fmnist_2class.yaml
├── experiments/             # 独立实验（含噪声扫描、消融对比等）
│   ├── noise_sweep.py
│   ├── depth_ablation.py
│   └── transferability.py
├── data/                    # 数据集（运行时下载）
├── logs/                    # 训练与攻击日志
├── results/                 # 实验结果（.json / .csv / .png）
└── docs/                    # 额外文档
    ├── ALGORITHM_NOTES.md   # 三个算法的原理笔记
    └── EXPERIMENT_GUIDE.md  # 实验流程与复现指南
```

## 环境依赖

推荐 Python 3.9+，显卡非必需（量子模拟全 CPU 也可，但建议 16GB+ 内存）。

```bash
pip install -r requirements.txt
```

核心依赖：

- `pennylane>=0.33` —— 量子机器学习框架，用于 PQC 构建与参数移位梯度
- `torch>=2.0` —— 经典部分（混合模型的经典分类头、优化器）
- `numpy`, `scipy`, `scikit-learn`
- `matplotlib`, `seaborn` —— 结果可视化
- `pyyaml` —— 配置管理
- `tqdm` —— 训练与攻击进度条

## 快速开始

### 1. 训练目标量子分类器

```bash
python scripts/train_classifier.py --config configs/mnist_2class.yaml
```

默认在 MNIST 16×16 上训练一个 10 层 PQC 二分类器，大约 5 分钟内结束。
训练完的模型权重保存在 `results/classifiers/`。

### 2. 运行 PSA 攻击（贡献 1）

```bash
python scripts/run_attack_psa.py \
    --classifier results/classifiers/mnist_2c_depth10.pth \
    --dataset mnist --epsilon 0.1 --steps 40
```

### 3. 运行 NAP 噪声感知攻击（贡献 2）

```bash
python scripts/run_attack_nap.py \
    --classifier results/classifiers/mnist_2c_depth10.pth \
    --noise-type depolarizing --noise-strength 0.05 --mc-samples 16
```

### 4. 运行 CMJA 跨模块联合攻击（贡献 3）

```bash
python scripts/run_attack_cmja.py \
    --config configs/mnist_2class.yaml \
    --alpha 0.5   # 量子/经典扰动权衡系数
```

完整实验流程与参数含义见 `docs/EXPERIMENT_GUIDE.md`。

## 研究状态

本项目为**研究原型**，结果仍在整理中，部分实验（如真实硬件上的测量、更大规模
数据集上的扩展）尚在推进。代码结构与实验脚本面向未来论文投稿组织。

## 致谢

项目设计参考了量子对抗攻击相关前期工作（Lu, Duan, Deng 2020；Liu & Wittek 2020；
Anil, Vinod, Narayan 2024 AAAI 等），在梯度估计与含噪模拟上调用了 PennyLane 的
parameter-shift 与 noise channel 接口。
