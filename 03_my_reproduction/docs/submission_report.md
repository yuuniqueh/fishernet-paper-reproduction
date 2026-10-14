# 实验总结

本项目用 PyTorch 实现 Deep FisherNet 的全图微调、局部特征提取、GMM 初始化、Fisher Layer 训练和 SVM 分类，并在 VOC2007 上测试 AlexNet、DINOv2-small 和 DINOv3 ViT-S/16。

## 测试结果

| 模型 | trainval → test mAP |
| --- | ---: |
| 论文 AlexNet（参考） | 83.40% |
| 论文 VGG16（参考） | 91.70% |
| 本项目 AlexNet | 78.22% |
| 本项目 DINOv2-small | **89.97%** |
| 本项目 DINOv3 ViT-S/16 | **87.67%** |

三个模型均训练全图 9,000 步、FisherNet 40,000 步。DINO 的图像尺度和模块学习率沿用项目配置，与论文原设置有差异。AlexNet 的两次历史实验在损失、初始化和 SPP 映射上也存在差异。

DINOv2 在本次 test 上比 DINOv3 高 2.30 个百分点；配对图像 bootstrap 的 95% 区间为 [1.37, 2.82] 个百分点。实验只使用种子 42。

## 训练前后的比较

| 模型 | 全图微调 + GMM 后 test mAP | FisherNet 40,000 步后 test mAP |
| --- | ---: | ---: |
| DINOv2-small | 89.40% | 89.97% |
| DINOv3 ViT-S/16 | 86.88% | 87.67% |

两个初始化对照在读取 test 指标前登记，正式模型固定使用最终 40,000 步 checkpoint。

独立 train/val 实验中，DINOv2 从初始化的 89.82% 降到 86.74%，DINOv3 从 85.24% 提高到 87.24%。验证集和测试集表现不同，不能据此把更多训练视为必然收益。完整曲线见[对照记录](backbone_comparison_20261013.md)。

## 数据与实现

验证实验使用 train 2,501 张、val 2,510 张；正式测试使用 trainval 5,011 张、test 4,952 张。各模型独立训练，网络、GMM 和 SVM 只使用对应训练集。

AlexNet 通过 SPP patch 提取局部描述子；DINO 使用空间 token，排除 CLS/register token。描述子投影到 256 维，经 32 分量 Fisher Layer 编码为 16,384 维向量，再进行 signed sqrt、L2 normalization 和线性 SVM 分类。

DINO 使用最长边 224/336 双尺度；训练实际步数、模块参数更新、GMM 和 SVM 收敛、预测 ID 与标签及 mAP 重算均已检查。代码测试最近一次为 78 项通过。

[运行说明](../README.md) · [结果文件](../results/README.md) · [论文](https://arxiv.org/abs/1608.00182)
