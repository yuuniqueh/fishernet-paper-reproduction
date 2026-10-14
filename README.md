# Deep FisherNet 论文复现

使用 PyTorch 实现 [Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182)，在 PASCAL VOC2007 上进行图像多标签分类。包含 AlexNet 实现，以及 DINOv2、DINOv3 的训练和测试结果。

流程：全图微调 → 局部特征 → GMM 初始化 → FisherNet 训练 → Fisher Vector + 线性 SVM。

## 结果

以下均为 trainval → test，指标为 VOC2007 11-point mAP。

| 模型 | mAP |
| --- | ---: |
| 论文 AlexNet（参考） | 83.40% |
| 论文 VGG16（参考） | 91.70% |
| 本项目 AlexNet | 78.22% |
| 本项目 DINOv2-small | **89.97%** |
| 本项目 DINOv3 ViT-S/16 | **87.67%** |

本项目三个模型均完成全图 9,000 步、FisherNet 40,000 步训练。DINO 使用 224/336 双尺度和较小学习率，与论文原设置有差异。上述结果来自单个训练种子。

DINOv2 在本次 test 上比 DINOv3 高 2.30 个百分点。全图微调后、尚未进行 FisherNet 更新的对照分别为 89.40% 和 86.88%。完整曲线和各类 AP 见[实验记录](03_my_reproduction/docs/backbone_comparison_20261013.md)。

## 使用

建议 Python 3.11、PyTorch 2.8。实验使用 RTX 4060 Laptop 8GB。

```powershell
cd 03_my_reproduction
python -m pip install -r requirements.txt
python -m pytest tests -q
```

VOC2007 放在 `04_dataset/VOC/VOCdevkit/VOC2007/`。DINOv2 另需 `requirements-backbones.txt`；DINOv3 可使用官方源码和本地权重。

配置、权重路径和训练命令见[运行说明](03_my_reproduction/README.md)。数据集、模型权重和原始日志保留在本地。

## 文件

- [运行说明](03_my_reproduction/README.md)
- [实验总结](03_my_reproduction/docs/submission_report.md)
- [DINOv2 / DINOv3 对照](03_my_reproduction/docs/backbone_comparison_20261013.md)
- [结果文件](03_my_reproduction/results/README.md)
