# Deep FisherNet 论文复现

本项目基于 PyTorch 复现论文 [Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182) 的核心方法，并在 PASCAL VOC2007 多标签分类任务上进行训练与评估。

## 项目内容

论文将卷积特征、Fisher Vector 与端到端训练结合。本实现完成了以下流程：

```text
VOC2007 图像
   ↓
AlexNet 全图微调
   ↓
局部 patch 特征 → GMM 拟合
   ↓
可学习 Fisher Layer
   ↓
多尺度 Fisher Vector → 线性 SVM → AP / mAP
```

| 模块 | 实现内容 |
| --- | --- |
| 特征提取 | ImageNet 预训练 AlexNet 与局部 patch 描述子 |
| 统计建模 | 32 分量 GMM、Fisher Vector 编码 |
| 端到端训练 | Fisher Layer 初始化、反向传播与学习率调度 |
| 多尺度推理 | 5 个图像尺度的特征汇聚 |
| 分类评估 | 线性 SVM、VOC2007 11-point AP 与 mAP |

## 核心设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | PASCAL VOC2007，20 类多标签分类 |
| 描述子维度 | 256 |
| GMM 分量数 | 32 |
| Fisher Vector 维度 | 16,384 |
| patch 尺寸 | 64、96、128、160、192、224、256 |
| 图像尺度 | 480、576、688、864、1200 |
| 归一化 | signed square root + L2 normalization |
| 分类器 | Linear SVM，`C=1` |

## 已完成工作

本项目不仅完成了模型搭建，还完成了从训练到评估的完整实验流程：

| 阶段 | 已完成内容 |
| --- | --- |
| 全图网络训练 | 基于 ImageNet 预训练 AlexNet 完成 9,000 步全图分类微调 |
| Fisher 初始化 | 从 1,000 张训练图像采样 99,928 个描述子，完成 32 分量 GMM 拟合 |
| FisherNet 训练 | 完成 40,000 步端到端训练，并保存训练诊断信息 |
| 多尺度评估 | 对 5 个尺度提取 Fisher Vector，使用 one-vs-all 线性 SVM 分类 |
| 独立验证 | 使用互不重叠的 train/val 划分；20 个类别的 SVM 均正常收敛 |
| 工程验证 | 提供数据读取、模型结构、GMM、训练和评估的核心单元测试 |

## 实验结果

| 实验 | 训练 / 评估划分 | mAP |
| --- | --- | ---: |
| 论文中 AlexNet 结果 | VOC2007 trainval / test | 83.4% |
| 本项目最佳基线 | VOC2007 trainval / test | **78.22%** |
| 独立验证实验 | VOC2007 train / val | 72.29% |

最佳基线使用五尺度 Fisher Vector 与线性 SVM。独立验证实验采用 train/val 划分，用于检查训练流程；由于划分不同，不与 trainval/test 指标直接比较。

### 最佳基线的分类别 AP

下表为 VOC2007 trainval/test 最佳基线（mAP 78.22%）的 20 类 AP：

| 类别 | AP | 类别 | AP |
| --- | ---: | --- | ---: |
| aeroplane | 89.12% | bicycle | 85.00% |
| bird | 87.93% | boat | 85.25% |
| bottle | 49.16% | bus | 72.56% |
| car | 89.09% | cat | 85.53% |
| chair | 66.25% | cow | 63.50% |
| diningtable | 73.03% | dog | 84.16% |
| horse | 87.35% | motorbike | 85.12% |
| person | 93.81% | pottedplant | 60.73% |
| sheep | 75.52% | sofa | 67.23% |
| train | 89.08% | tvmonitor | 74.91% |

完整的结果记录、权重标识和独立验证结果见 [结果摘要](03_my_reproduction/results/verified_results.json)。

## 实验流程

```text
ImageNet 预训练权重
        ↓
VOC2007 全图分类微调
        ↓
训练集局部描述子采样与 GMM 初始化
        ↓
FisherNet 端到端训练（40,000 steps）
        ↓
五尺度 Fisher Vector 提取与 SVM 评估
```

## 工程结构

```text
03_my_reproduction/
├── configs/       YAML 实验配置
├── docs/          实验报告与实现说明
├── models/        AlexNet、SPP 与 FisherNet 网络定义
├── results/       可核验的结果摘要
├── src/           数据处理、GMM、训练和评估工具
├── tests/         单元测试
├── train_whole_image.py      全图分类微调
├── initialize_fisher.py      GMM 拟合与 FisherNet 初始化
├── train.py                  FisherNet 训练
└── evaluate.py               SVM 与 mAP 评估
```

## 快速开始

```powershell
cd 03_my_reproduction
python -m pip install -r requirements.txt
python -m pytest tests -q
```

将 VOC2007 放入 `04_dataset/VOC/VOCdevkit/VOC2007/` 后，依次运行：

```text
train_whole_image.py → initialize_fisher.py → train.py → evaluate.py
```

完整命令、数据目录和配置说明见 [实现说明](03_my_reproduction/README.md)。数据集、预训练权重和训练产生的 checkpoint 因体积原因未纳入仓库。

## 相关文档

- [实现说明](03_my_reproduction/README.md)
- [实验总结](03_my_reproduction/docs/submission_report.md)
- [结果摘要](03_my_reproduction/results/verified_results.json)
