# FisherNet 实现说明

本目录包含 Deep FisherNet 的 PyTorch 实现、配置文件、训练脚本与测试代码。

## 环境与数据

建议使用 Python 3.10 及 CUDA 环境。安装依赖并执行测试：

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
```

VOC2007 数据集目录如下：

```text
04_dataset/VOC/VOCdevkit/VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/Main/
    ├── train.txt
    ├── val.txt
    ├── trainval.txt
    └── test.txt
```

## 代码说明

| 文件或目录 | 功能 |
| --- | --- |
| `configs/` | YAML 实验配置，包括数据、模型和训练参数 |
| `models/` | AlexNet、SPP 与 FisherNet 网络结构 |
| `src/` | 数据读取、patch 生成、GMM、Fisher 编码及公共工具 |
| `tests/` | 数据、模型、训练和评估的单元测试 |
| `train_whole_image.py` | 对全图分类网络进行微调 |
| `initialize_fisher.py` | 提取描述子、拟合 GMM、初始化 FisherNet |
| `train.py` | FisherNet 端到端训练 |
| `evaluate.py` | 提取多尺度 Fisher Vector，训练 SVM 并计算 mAP |
| `evaluate_whole_image.py` | 评估全图分类微调模型 |

## 训练流程

### 1. 全图分类微调

```powershell
python train_whole_image.py --config configs/voc2007_alexnet.yaml --output checkpoints/whole_image.pt
```

### 2. GMM 拟合与 FisherNet 初始化

```powershell
python initialize_fisher.py --config configs/voc2007_alexnet.yaml --whole-image-checkpoint checkpoints/whole_image.pt --output checkpoints/fishernet_gmm.pt
```

### 3. FisherNet 训练

```powershell
python train.py --config configs/voc2007_alexnet.yaml --init-checkpoint checkpoints/fishernet_gmm.pt --output checkpoints/fishernet.pt
```

### 4. 多尺度测试与 SVM 评估

```powershell
python evaluate.py --config configs/voc2007_alexnet.yaml --checkpoint checkpoints/fishernet.pt --protocol svm --split test --fit-split trainval --output-dir ../05_results/evaluation
```

## 主要配置

`configs/voc2007_alexnet.yaml` 是 VOC2007 trainval/test 基线配置。核心参数为：7 种 patch 尺寸、5 个推理尺度、256 维描述子、32 分量 GMM、16,384 维 Fisher Vector 以及 `C=1` 的线性 SVM。

该配置对应本项目已保存的最佳结果：VOC2007 test mAP 为 **78.22%**。

## 输出文件

训练会生成 checkpoint；评估目录中会生成：

- `voc2007_svm_metrics.json`：各类别 AP 与总 mAP；
- `voc2007_svm_predictions.npz`：预测分数和标签；
- 日志及中间结果文件。

数据集、ImageNet 预训练权重和训练生成的 checkpoint 未提交到仓库。

## 参考资料

- [实验总结](docs/submission_report.md)
- [结果摘要](results/verified_results.json)
