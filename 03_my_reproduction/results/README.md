# 结果文件

指标为 VOC2007 图像多标签分类的 11-point mAP；JSON 中 AP 和 mAP 的取值为 0～1。

| 模型 | test mAP | trainval/test 训练预算 |
| --- | ---: | --- |
| AlexNet | 78.22% | 全图 9,000 + Fisher 40,000 步 |
| DINOv2-small | 89.97% | 全图 9,000 + Fisher 40,000 步 |
| DINOv3 ViT-S/16 | 87.67% | 全图 9,000 + Fisher 40,000 步 |

DINOv2、DINOv3 初始化模型的 test mAP 分别为 89.40%、86.88%；它们只完成全图微调和 GMM 初始化。

| 文件 | 内容 |
| --- | --- |
| [verified_results.json](verified_results.json) | 各实验的指标摘要 |
| [backbone_comparison_20261013.json](backbone_comparison_20261013.json) | DINOv2 / DINOv3 的配置、训练曲线、test 结果、各类 AP 和预测核验 |
| [backbone_trials_20261011.json](backbone_trials_20261011.json) | 冻结特征的 train/val 对照 |
| [dinov3_finetune_20261011.json](dinov3_finetune_20261011.json) | DINOv3 短程训练 |
| [dinov3_40000_20261011.json](dinov3_40000_20261011.json) | DINOv3 完整预算的 train/val 结果 |

val 与 test 使用不同划分，分别报告。原始模型、逐图预测和日志保留在本地。

[实验记录](../docs/backbone_comparison_20261013.md) · [运行说明](../README.md)
