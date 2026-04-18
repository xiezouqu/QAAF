# 算法原理笔记

这份笔记把 QAAF 三个核心算法的数学动机和实现细节串起来，供答辩/讲解时参考。

---

## 1. 参数移位规则与输入空间对抗攻击

### 1.1 经典对抗攻击的数学形式

对一个分类器 $f_\theta : \mathbb{R}^d \to \mathbb{R}^K$ 和输入 $(x, y)$，
标准的对抗攻击通过求解如下约束优化问题：

$$
\max_{\|\delta\|_p \le \epsilon} \;\; \mathcal{L}(f_\theta(x + \delta),\, y)
$$

其中 $\mathcal{L}$ 通常是交叉熵损失，$\|\cdot\|_p$ 是范数约束（常取 $p=\infty$ 或 $p=2$），
$\epsilon$ 是扰动半径。FGSM、PGD、MIFGSM 等方法都是这个问题的不同优化变体。

### 1.2 量子分类器带来的困难

参数化量子电路分类器 $f_\theta$ 的前向过程是：

$$
x \xrightarrow{\text{振幅编码}} |\psi_x\rangle \xrightarrow{U(\theta)} |\phi\rangle
\xrightarrow{\text{测量}} P(\hat c = k \mid x)
$$

问题在于：**量子测量会让态坍缩，所以没法像经典神经网络那样反向传播。**
因此经典的 FGSM/PGD 里的 $\nabla_x \mathcal{L}$ 必须换一种方式计算。

### 1.3 参数移位规则 (PSR)

对形如 $U(\theta) = \exp(-i\theta G / 2)$ 的量子门（$G$ 本征值为 $\pm 1$），有恒等式：

$$
\frac{\partial \langle \mathcal{L} \rangle}{\partial \theta}
= \tfrac{1}{2} \left[
  \langle \mathcal{L} \rangle_{\theta + \pi/2} - \langle \mathcal{L} \rangle_{\theta - \pi/2}
\right]
$$

这是**精确**的梯度表达式，不是有限差分近似。在硬件上只需两次额外的电路评估。

### 1.4 输入空间梯度怎么办

注意 PSR 直接针对的是对**参数** $\theta$ 的梯度。我们要的是对**输入** $x$ 的梯度。
有两条路：

**路线 A（仿真）**：振幅编码 $x \to |\psi_x\rangle$ 本身就是一个可微映射，
PennyLane 的 autograd 能穿过整个电路直接给出 $\nabla_x \mathcal{L}$。
这是 `analytic_input_gradient` 走的路，速度快但只限于模拟器。

**路线 B（真机）**：把每个输入维度 $x_i$ 当作一个"伪参数"，用中心差分：

$$
\frac{\partial \mathcal{L}}{\partial x_i}
\approx \frac{\mathcal{L}(x + \varepsilon e_i) - \mathcal{L}(x - \varepsilon e_i)}{2\varepsilon}
$$

这是 `finite_difference_gradient` 做的事。每维需要两次电路评估，
总共 $2d$ 次。对 16×16 MNIST 就是 512 次。

### 1.5 PSA 实现中的选择

`ParameterShiftAttack` 支持三种变体：

- **FGSM**（单步）：$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$
- **PGD**（多步）：迭代 FGSM，每步后投影回 $\epsilon$ 邻域
- **MIFGSM**（带动量）：$m_{t+1} = \mu m_t + \frac{g_t}{\|g_t\|_1}$，再做一步

PGD 是默认选择，因为单步 FGSM 会 overshoot，而 MIFGSM 主要用于迁移攻击实验。

---

## 2. 噪声感知扰动优化 (NAP)

### 2.1 为什么需要噪声感知

NISQ 硬件的去极化噪声会让每次测量期望 $\langle \mathcal{L} \rangle$ 带有涨落。
如果把 PSA 直接搬到含噪设备上，梯度估计本身就不稳：

$$
\hat{g} = \nabla_x \mathcal{L}_{\text{noisy}}(x) = \nabla_x \mathcal{L}(x) + \eta
$$

其中 $\eta$ 是噪声引入的随机项。基于 $\hat{g}$ 做符号操作 $\text{sign}(\hat{g})$ 时，
小梯度分量的符号很容易被 $\eta$ 翻转，导致扰动方向偏离最优。

### 2.2 蒙特卡洛期望化

NAP 的解法是：在每一步迭代对含噪梯度做多次采样取平均。

$$
\bar{g} = \frac{1}{M} \sum_{m=1}^{M} \nabla_x \mathcal{L}_{\text{noisy}}^{(m)}(x)
\xrightarrow{M \to \infty} \mathbb{E}_\eta [\nabla_x \mathcal{L}_{\text{noisy}}(x)]
$$

由于我们关心的去极化信道是线性的，期望可以穿过求导：

$$
\mathbb{E}[\nabla_x \mathcal{L}_{\text{noisy}}] = \nabla_x \mathbb{E}[\mathcal{L}_{\text{noisy}}]
$$

所以 $\bar{g}$ 实际上就是对**期望损失**的梯度。

### 2.3 样本数 M 怎么选

标准差按 $1/\sqrt{M}$ 下降，所以 $M$ 不用很大就能显著稳定。
实验表明 $M = 8 \sim 32$ 在噪声强度 $p \le 0.1$ 的情况下已经足够。
`NAPConfig.anneal_schedule` 支持退火：前几步 $M$ 小（探索），
后面 $M$ 大（精修）。

### 2.4 与直接攻击含噪模型的区别

很重要的一点：NAP 不是"给含噪模型做攻击"，它是"用噪声感知的方式给任意模型做攻击"。
即使模型本身是干净的，只要我们预期它会部署到含噪硬件上，用 NAP 生成的扰动在实际
硬件上会比 PSA 扰动更稳定。

---

## 3. 量子-经典混合模型的跨模块联合攻击 (CMJA)

### 3.1 混合模型结构

实际 NISQ 部署常见的架构：

$$
x \xrightarrow{\text{PQC} = f_\theta^{(q)}} \phi(x) \in [0,1]^{2^K}
\xrightarrow{\text{MLP} = f_\omega^{(c)}} \text{logits}
$$

量子部分负责特征提取，经典 MLP 负责分类。这样辅助 qubit 数 $K$ 不需要随类别数
指数增长，节省硬件资源。

### 3.2 两个脆弱面

在这种架构里有两个攻击者可以施力的地方：

- **量子端**：扰动 $x$ 让 $\phi(x)$ 偏离正常分布
- **经典端**：即使 $\phi(x)$ 正常，经典 MLP 的决策边界也可能被 $\phi$ 空间的
  小扰动翻越

单独攻击量子端或经典端都可能次优，因为：
- 纯量子攻击忽略了经典 MLP 的具体决策边界
- 纯经典攻击无法真正修改 $\phi$（$\phi$ 是 $x$ 决定的，不是独立输入）

### 3.3 联合攻击的两路梯度

CMJA 计算两个梯度：

$$
g_x = \nabla_x \mathcal{L}(f^{(c)}(\phi(x)), y) \qquad \text{(直接路径)}
$$

$$
g_{\phi \to x} = \left( \frac{\partial \phi}{\partial x} \right)^\top \nabla_\phi \mathcal{L}
\qquad \text{(经由特征空间的路径)}
$$

$g_x$ 是标准的输入空间梯度，$g_{\phi \to x}$ 是"用经典分类头的信息指导输入扰动"。

### 3.4 组合权重

用超参数 $\lambda$ 组合两个方向：

$$
\text{direction} = \lambda \cdot \text{sign}(g_x) + (1 - \lambda) \cdot \text{sign}(g_{\phi \to x})
$$

- $\lambda = 1$：退化为 PSA（纯量子输入攻击）
- $\lambda = 0$：退化为"完全通过 $\phi$ 反传的攻击"
- $\lambda = 0.5$：均衡的联合攻击

我们通过消融实验（`mode` 参数）验证 $\lambda \in (0, 1)$ 的联合方式优于两个极端。

### 3.5 实现中的近似

理论上 $g_{\phi \to x}$ 需要显式算 Jacobian $\partial \phi / \partial x$，
这是一个 $(2^K \times d)$ 的矩阵。我们用**数值 JVP**（Jacobian-vector product）
直接算：

$$
[g_{\phi \to x}]_i \approx \frac{\phi(x + \varepsilon e_i) - \phi(x)}{\varepsilon} \cdot g_\phi
$$

复杂度 $O(d \cdot \text{PQC forward})$，避免了 $O(d \cdot 2^K)$ 的显式 Jacobian。

---

## 4. 实现上的几个关键决策

### 4.1 为什么默认用 `default.qubit` 而不是 `default.mixed`

- `default.qubit` 速度快，支持 backprop，但不支持噪声信道
- `default.mixed` 支持密度矩阵 + Kraus 算子，是含噪仿真必备

所以代码里的逻辑是：只要 `noise` 非空就自动切到 `default.mixed`。
训练阶段通常不含噪，用 `default.qubit` + backprop；攻击阶段按需切换。

### 4.2 为什么分类器用 `n_anc_qubits` 而不是直接测 data qubits

这是 QML 里的常见技巧：用辅助 qubit 读出标签可以让数据 qubit 保持"未测量"状态，
便于做 ensemble 或直接复用同一个电路提取多类概率。辅助 qubit 数 $K = \lceil \log_2 C \rceil$
就能支持 $C$ 类分类。

### 4.3 为什么训练用 numpy + autograd 而不是 PyTorch

- PennyLane 的 `default.qubit` + autograd 组合原生支持 `qml.grad`，不需要额外的
  接口转换
- 对于 PQC 这种"层少、参数多、前向慢"的模型，PyTorch 的优势（大 batch、CUDA）
  用不太上
- 经典分类头（`HybridClassifier.head`）还是用 PyTorch，因为它就是标准 MLP

### 4.4 为什么参数初始化用小 scale

barren plateau 问题：对随机初始化的深层 PQC，损失函数关于参数的梯度期望指数级
接近 0，训练卡住。小初始化（$\text{scale} = 0.1$）让初始参数都集中在单位元附近，
梯度信号较强，早期训练稳定。

---

## 5. 与相关工作的定位

### 5.1 和 (Lu, Duan, Deng 2020) 的关系

他们最先在 Physical Review Research 上展示了量子分类器对对抗样本的敏感性，
但**没有给出可复用的算法框架**，实验局限在小规模。
QAAF 提供了一个模块化实现，并在 NAP 这个维度上做了扩展。

### 5.2 和 (Gong, Deng 2022) 的关系

他们理论上证明了量子 UAP 的存在，并给出 qBIM 迭代算法。我们的 PSA 关注的是
**输入特异性**对抗样本（每个输入对应不同扰动），与 UAP 互补。

### 5.3 和 QuGAP (AAAI 2024) 的关系

QuGAP 做**通用对抗扰动**，用生成模型产生 UAP。我们的三类攻击都是
**输入特异性**的，适合不同的威胁模型：
- QuGAP：攻击者一次训练好，批量部署（input-agnostic）
- QAAF：攻击者针对每个具体目标样本做优化（input-specific）

### 5.4 本研究的独特定位

QAAF 是面向**NISQ 实际部署场景**设计的：
- NAP 关注**硬件噪声下攻击稳定性**（其他工作假设无噪或只测一下噪声下鲁棒性）
- CMJA 关注**量子-经典混合架构**（其他工作多针对纯量子分类器）
- PSA 实现了**硬件兼容的梯度估计**（finite_difference 模式可在真机上跑）

---

## 6. 局限与未来方向

- 仿真规模局限：目前最大到 16×16 MNIST（8 qubit），更大规模（如 32×32）
  需要真机或更高效的模拟器
- NAP 的 MC 开销：$M$ 次评估会线性放大时间，实际硬件成本很高
- CMJA 的 JVP 数值稳定性：$\varepsilon$ 太小会放大噪声，太大会引入截断误差
- 防御机制尚未探索：本工作聚焦"攻击"，相应的防御（对抗训练、随机编码等）
  留作后续研究
