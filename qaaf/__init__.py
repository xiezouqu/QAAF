"""
QAAF: Quantum Adversarial Attack Framework
================================================================================

面向参数化量子电路分类器的对抗攻击研究原型。

核心模块:
    circuits        - PQC 分类器结构、振幅编码、含噪电路的构造
    classifiers     - 量子分类器、量子-经典混合分类器包装类
    gradient        - 基于参数移位规则的梯度估计
    attacks         - 三类攻击实现 (PSA / NAP / CMJA)
    noise_models    - 去极化、比特翻转、振幅阻尼等噪声信道
    datasets        - 经典数据集的下采样、归一化与振幅编码预处理
    metrics         - 攻击成功率、扰动幅度、迁移率等评估指标
    utils           - 通用工具函数

典型使用流程:
    >>> from qaaf import classifiers, attacks, datasets
    >>> clf = classifiers.PQCClassifier(n_qubits=9, n_layers=10)
    >>> X, y = datasets.load_mnist_amplitude(n_class=2, size=16)
    >>> clf.fit(X, y, epochs=20)
    >>> atk = attacks.ParameterShiftAttack(clf, epsilon=0.1)
    >>> X_adv = atk.generate(X)
    >>> success_rate = (clf.predict(X_adv) != y).mean()
"""

__version__ = "0.1.0"
__author__ = "QAAF Project"

from qaaf import circuits
from qaaf import classifiers
from qaaf import gradient
from qaaf import attacks
from qaaf import noise_models
from qaaf import datasets
from qaaf import metrics
from qaaf import utils

__all__ = [
    "circuits",
    "classifiers",
    "gradient",
    "attacks",
    "noise_models",
    "datasets",
    "metrics",
    "utils",
]
