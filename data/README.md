# data/

首次运行 `scripts/train_classifier.py` 时会自动下载 MNIST 或 Fashion-MNIST 到
这个目录（通过 torchvision）。所以你不需要手动放任何数据文件在这里。

合成数据（`datasets.load_synthetic`）不需要下载，直接在内存里生成。
