"""
通用工具函数模块。

包含:
    - 随机种子控制（保证实验可复现）
    - 日志管理（控制台 + 文件双输出）
    - 模型/结果序列化
    - 装饰器工具（计时、缓存）
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 随机性控制
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置所有涉及随机性的库的种子，保证实验可复现。

    Parameters
    ----------
    seed : int
        全局随机种子。
    deterministic : bool
        是否要求 CUDA 算子完全确定（会牺牲一些速度）。

    Notes
    -----
    即便设了所有种子，PennyLane 的 shot-based 模拟在实际硬件后端上仍会因测量
    本身的统计涨落产生非确定性结果。在纯 analytic 模拟下（shots=None）可完全复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    构造一个带控制台 + 可选文件输出的 logger。

    已经配置过的 logger 不会重复添加 handler，所以可以反复调用同名 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def save_json(obj: Dict[str, Any], path: str) -> None:
    """把字典保存为 JSON（自动处理 numpy 类型）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 装饰器
# ---------------------------------------------------------------------------

def timeit(func: Callable) -> Callable:
    """简单的函数计时装饰器。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        # 尝试从 kwargs 找 logger，否则打印
        logger = kwargs.get("logger", None)
        msg = f"[timeit] {func.__name__} took {dt:.3f}s"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return result

    return wrapper


def ensure_dir(path: str) -> str:
    """确保目录存在，返回原路径。"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------

def normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对向量做 L2 归一化，避免除零。"""
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm


def clip_perturbation(
    delta: np.ndarray,
    epsilon: float,
    norm: str = "linf",
) -> np.ndarray:
    """
    将扰动投影到 Lp-球内。

    Parameters
    ----------
    delta : ndarray
        原始扰动。
    epsilon : float
        扰动半径。
    norm : {'linf', 'l2'}
        范数类型。

    Returns
    -------
    ndarray
        截断后的扰动。
    """
    if norm == "linf":
        return np.clip(delta, -epsilon, epsilon)
    elif norm == "l2":
        n = np.linalg.norm(delta)
        if n > epsilon:
            return delta * (epsilon / n)
        return delta
    else:
        raise ValueError(f"Unknown norm type: {norm}")


def clip_image(x: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """把图像像素裁剪到合法范围（通常 [0, 1]）。"""
    return np.clip(x, low, high)
