# results/

运行时产生的所有实验结果存放于此：

- `classifiers/` — 训练好的分类器权重（.pkl / .pt）与训练信息（.info.json）
- `attacks/` — 各种攻击的 JSON 报告与对抗样本（.npz）
- `experiments/` — 独立实验（noise_sweep, depth_ablation, transferability）的汇总结果
- `figures/` — 可视化图（.png）

这些文件在 Git 中被忽略，只在本地复现实验时生成。
