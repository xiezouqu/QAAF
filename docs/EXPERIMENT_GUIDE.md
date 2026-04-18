# 实验流程与复现指南

本文档给出 QAAF 所有核心实验的完整复现流程。所有命令都假设当前目录是项目根
(`QAAF/`)，脚本从这里启动。

## 0. 环境检查

```bash
# 安装依赖
pip install -r requirements.txt

# 确认 PennyLane 可用
python -c "import pennylane as qml; print(qml.about())"
```

如果 PennyLane 显示 Lightning 后端可用，单次电路模拟速度会快一倍以上。

## 1. 快速自检：合成数据跑通全流程

在动用真实数据集前，先用合成数据确认代码链路没问题：

```bash
# 训练一个非常小的 PQC
python scripts/train_classifier.py \
    --dataset synthetic --classes 0,1 \
    --image-size 4 --n-per-class 80 \
    --depth 4 --epochs 10 \
    --tag synthetic_smoke

# 运行 PSA
python scripts/run_attack_psa.py \
    --classifier results/classifiers/synthetic_smoke.pkl \
    --dataset synthetic --classes 0,1 \
    --image-size 4 --n-test 20 \
    --variant pgd --epsilon 0.1 --steps 10 \
    --output results/attacks/smoke_psa.json
```

在普通 CPU 机器上应该 2-3 分钟跑完。看到 `attack_success_rate` 明显大于 0 就说明
链路通了。

## 2. 主实验 A：贡献 1 (PSA) 在 MNIST 上的系统评估

### 2.1 训练目标分类器

```bash
python scripts/train_classifier.py --config configs/mnist_2class.yaml
```

产出：
- `results/classifiers/mnist_2c_d10.pkl` 模型权重
- `results/classifiers/mnist_2c_d10.info.json` 训练信息

预期 `test_acc` 在 `0.92 - 0.98` 之间。

### 2.2 扫描扰动强度

对不同 $\epsilon$ 分别跑 PSA-PGD：

```bash
for EPS in 0.05 0.08 0.10 0.12 0.15 0.20; do
    python scripts/run_attack_psa.py \
        --classifier results/classifiers/mnist_2c_d10.pkl \
        --variant pgd --epsilon $EPS --steps 40 \
        --output results/attacks/psa_eps${EPS}.json
done
```

### 2.3 绘制曲线

```bash
python scripts/plot_results.py \
    --kind psa \
    --pattern "results/attacks/psa_eps*.json" \
    --output results/figures/psa_asr_vs_eps.png
```

### 2.4 可视化几个对抗样本

```bash
python scripts/plot_results.py \
    --kind samples \
    --input results/attacks/psa_eps0.10.npz \
    --output results/figures/psa_samples.png
```

## 3. 主实验 B：贡献 2 (NAP) 在不同噪声下的稳定性

### 3.1 噪声强度扫描

```bash
python experiments/noise_sweep.py \
    --classifier results/classifiers/mnist_2c_d10.pkl \
    --noise-type depolarizing \
    --noise-values 0.0 0.01 0.03 0.05 0.08 0.10 \
    --mc-samples 8 \
    --epsilon 0.10 --steps 30
```

产出 `results/experiments/noise_sweep.json`，其中包含每个噪声强度下 PSA vs NAP
的对比数据。

### 3.2 绘图对比

`noise_sweep.json` 的结构是 `{"psa": [...], "nap": [...]}`，手动或用一个小脚本
绘制两条曲线即可。示例：

```bash
python -c "
import json, matplotlib.pyplot as plt
r = json.load(open('results/experiments/noise_sweep.json'))
noise = r['noise_values']
psa_asr = [e['attack_success_rate'] for e in r['psa']]
nap_asr = [e['attack_success_rate'] for e in r['nap']]
plt.plot(noise, psa_asr, 'o-', label='PSA (noise-unaware)')
plt.plot(noise, nap_asr, 's-', label='NAP (noise-aware)')
plt.xlabel('Depolarizing noise strength')
plt.ylabel('Attack Success Rate')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('results/figures/nap_vs_psa.png', dpi=150)
"
```

预期结果：noise = 0 时 PSA/NAP 几乎相同；noise 升高时 NAP 的 ASR 衰减更缓。

## 4. 主实验 C：贡献 3 (CMJA) 在混合模型上的对比

### 4.1 一键训练 + 攻击

`run_attack_cmja.py` 如果没传 `--classifier`，会自动训练一个混合模型再攻击：

```bash
python scripts/run_attack_cmja.py \
    --dataset mnist --classes 0,1,2,3 \
    --image-size 16 --depth 6 \
    --n-per-class-train 120 --n-test 40 \
    --epsilon 0.10 --steps 40 --lam 0.5
```

产出：
- `results/classifiers/hybrid_mnist_4c_d6.pt` 混合模型
- `results/attacks/cmja_result.json` 三种模式的对比报告
- `results/attacks/cmja_joint.npz` / `cmja_quantum_only.npz` / `cmja_classical_only.npz`
  三组对抗样本

### 4.2 绘制柱状对比

```bash
python scripts/plot_results.py \
    --kind cmja \
    --input results/attacks/cmja_result.json \
    --output results/figures/cmja_modes.png
```

### 4.3 扫描 lambda

想看最优 $\lambda$，循环跑几次（这个脚本默认会覆盖 joint 模式的输出，要自己加 tag）：

```bash
for L in 0.2 0.4 0.5 0.6 0.8; do
    python scripts/run_attack_cmja.py \
        --classifier results/classifiers/hybrid_mnist_4c_d6.pt \
        --lam $L \
        --output results/attacks/cmja_lam${L}.json
done
```

## 5. 独立实验：电路深度消融

```bash
python experiments/depth_ablation.py \
    --dataset mnist --classes 0,1 \
    --depths 4 8 12 16 \
    --epochs 15 --n-per-class 150 \
    --epsilon 0.10 --steps 30
```

产出 `results/experiments/depth_ablation.json`。

## 6. 独立实验：攻击迁移性

先训几个不同 seed / 不同深度的 source 和 target：

```bash
for SEED in 42 0 7; do
    python scripts/train_classifier.py \
        --config configs/mnist_2class.yaml \
        --seed $SEED --tag mnist_2c_d10_s${SEED}
done

for DEPTH in 6 10 14; do
    python scripts/train_classifier.py \
        --config configs/mnist_2class.yaml \
        --depth $DEPTH --tag mnist_2c_d${DEPTH}
done
```

然后跑迁移性测试：

```bash
python experiments/transferability.py \
    --source results/classifiers/mnist_2c_d10_s42.pkl \
    --targets \
        results/classifiers/mnist_2c_d10_s0.pkl \
        results/classifiers/mnist_2c_d10_s7.pkl \
        results/classifiers/mnist_2c_d6.pkl \
        results/classifiers/mnist_2c_d14.pkl \
    --variant mifgsm --epsilon 0.15
```

预期结果：同架构不同 seed 的迁移率最高（60%+），
不同深度的迁移率递减（30-50%）。

## 7. 性能与计算资源

**单次实验参考耗时（单核 CPU）：**

| 实验 | 样本数 | 层数 | 大致耗时 |
|------|--------|------|----------|
| PSA-PGD（单 epsilon） | 50 | 10 | ~3 分钟 |
| NAP（MC=8） | 30 | 10 | ~8-15 分钟 |
| CMJA（3 模式） | 40 | 6 | ~10-20 分钟 |
| 训练分类器 | 400 train | 10 | ~5 分钟 |
| 噪声扫描（6 个点） | 20 × 6 | 10 | ~30-50 分钟 |

**加速建议：**

1. 装 PennyLane-Lightning 后端，单次电路快 2-3 倍：
   ```bash
   pip install pennylane-lightning
   ```
   然后在 `classifiers.PQCClassifier.__init__` 或 `build_classifier_qnode`
   里把 `device_name` 改成 `"lightning.qubit"`（无噪场景下）。

2. 如果有 GPU，`default.mixed` 可通过 JAX interface 获得加速，需要把
   `interface="autograd"` 改成 `"jax"`。

3. 实验时优先跑小规模（`n_per_class=50`）验证方向对了，再放大。

## 8. 常见问题

**Q: PQC 训练完 accuracy 还是接近 0.5？**
A: 多半是初始化太大落到了 barren plateau。检查 `circuits.init_classifier_weights`
   的 `scale` 参数，默认是 0.1，如果你改成 1.0 就会出问题。

**Q: NAP 报 pennylane error 或 noise channel 不可用？**
A: `default.qubit` 不支持噪声信道，需要用 `default.mixed`。代码里已经做了自动切换，
   但如果你手动指定了 `device_name="default.qubit"` + 非零噪声，会报错。

**Q: CMJA 的 JVP 估计巨慢？**
A: 对 256 维输入，JVP 需要 256 次 PQC 前向，一个 step 就要几百次电路评估。
   建议：在原型实验上用 `image_size=8`（64 维），主实验再扩到 16。

**Q: 如何在真机上跑？**
A: 改 `device_name` 为对应厂商的后端（如 `"qiskit.ibmq"`），并把 `diff_method`
   切到 `"parameter-shift"`，`shots` 设具体数。详见 PennyLane 文档。
