# Deep FisherNet 复现实验报告

## 1. 项目目标

使用 PyTorch 实现论文《Deep FisherNet for Object Classification》的主要流程，并在 PASCAL VOC2007 多标签分类任务上完成训练和评估。

## 2. 实现内容

本项目完成了以下模块：

1. 基于 ImageNet 预训练 AlexNet 的全图分类微调；
2. 局部 patch 描述子提取与 GMM 初始化；
3. 可学习 Fisher Layer 与端到端训练；
4. 五尺度 Fisher Vector 融合；
5. 线性 SVM 分类、VOC2007 AP 和 mAP 计算。

主要设置为 256 维描述子、32 分量 GMM、16,384 维 Fisher Vector、7 种 patch 尺寸、5 个图像尺度，以及 `C=1` 的线性 SVM。

## 3. 实验结果

| 实验 | 训练 / 评估划分 | mAP |
| --- | --- | ---: |
| 论文 AlexNet 参考结果 | VOC2007 trainval / test | 83.40% |
| 本项目最佳基线 | VOC2007 trainval / test | **78.22%** |
| 独立验证实验 | VOC2007 train / val | 72.29% |

最佳基线完成 40,000 步 FisherNet 训练，并在 VOC2007 test 的 4,952 张图像上评估。独立验证实验使用 2,501 张训练图像和 2,510 张验证图像，20 个类别的 SVM 均收敛。

完整的 20 类 AP 与实验摘要见 [结果文件](../results/verified_results.json)。

## 4. 工程说明

仓库包含训练、初始化、评估脚本和核心单元测试。标准运行流程为：

```text
全图分类微调 → GMM 初始化 → FisherNet 训练 → 多尺度 SVM 评估
```

VOC2007 数据集、预训练权重和训练 checkpoint 因体积原因未提交；其余代码、配置和结果摘要均已包含在仓库中。
