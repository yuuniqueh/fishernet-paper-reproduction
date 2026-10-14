# DINOv3 短程训练与验证（2026-10-11）

本次使用用户已有的官方 DINOv3 ViT-S/16 权重，完成全图训练、GMM 初始化、
FisherNet 端到端训练及独立验证。`freeze_backbone: false`，DINOv3 权重参与两个阶段的反向传播和更新。
这次是实际短程训练试验，区别于此前冻结特征的 PCA/GMM/SVM 试验。
这里完成的是短程试验的全部流程，并未达到原 AlexNet 的 9,000 + 40,000 步训练预算。
5,000 步不是根据收敛证据确定的，不能据此判断 DINOv3 充分训练后的性能。
后续已完成独立的 9,000 + 40,000 步实验，最终 val mAP 87.24%；
详见 [40,000 步实验记录](dinov3_40000_20261011.md)。下文保留本次短程试验的原始结果。

## 结果

全部使用 VOC2007 train（2,501 张）训练，val（2,510 张）评估，VOC2007 11-point AP。
没有使用 test。各类别 AP、预测校验、参数更新审计及配对 bootstrap 结果见
[完整结果](../results/dinov3_finetune_20261011.json)。

| 模型 / 阶段 | 分类与推理协议 | val mAP |
| --- | --- | ---: |
| 已有 AlexNet 独立验证 | 原五尺度 Fisher + SVM | 72.29% |
| DINOv3：全图训练 1,500 步 | 单中心裁剪、全图分类头 | 49.85% |
| DINOv3：全图训练 + GMM，FisherNet 更新前 | 224/336 双尺度 Fisher + SVM | **86.82%** |
| DINOv3：FisherNet 端到端训练 5,000 步后 | 224/336 双尺度 Fisher + SVM | 86.30% |
| DINOv3：FisherNet 端到端训练 5,000 步后 | 单中心裁剪、FisherNet 自带分类头 | 63.60% |

训练后的 Fisher+SVM 比历史 AlexNet 验证结果高 **14.01 个百分点**，
但比相同提取和 SVM 配置下、FisherNet 更新前的 86.82% 低 **0.51 个百分点**。
因此，本次新的 DINOv3 方案得到更高的验证性能，但当前 5,000 步 FisherNet 更新没有带来额外收益。
两份 checkpoint 均保留；本次验证集表现较好的候选是 `fisher_gmm.pt`。
它已经经过全图训练和 GMM 拟合，仅尚未进行 FisherNet 端到端更新。

AlexNet 和 DINOv3 的描述子、尺度与训练步数不同，这不是仅替换 backbone 的严格单变量消融。
此前冻结 DINOv3 的 85.27% 则使用 PCA 64 维、16 分量 GMM 和 224×224 整图缩放，
也不能把两次试验之间的差值全部解释为微调收益。

自带分类头与另行拟合的 SVM 是不同预测路线。这里的 86.30% 对应 Fisher+SVM，
直接调用 FisherNet 分类头得到的验证指标是 63.60%。

## 实验设置

- 权重：`D:/enter_project/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`。
- 官方源码：`D:/enter_project/third_party/dinov3-main`；直接离线、严格加载本地 state dict。
- Backbone：ViT-S/16，384 维空间 token；CLS/register token 不进入 Fisher 聚合。
- 描述子投影：384→256 维可学习线性层；GMM 32 个对角协方差分量，Fisher Vector 16,384 维。
- 全图训练：1,500 步、batch 16、224 像素中心裁剪和随机翻转。
- GMM：从 1,000 张 train 图像均匀采样 50,000 个描述子，9 次 EM 迭代后收敛。
- FisherNet：5,000 步、微批次 1、梯度累积 2，随机最长边尺度 224/336，保留图像宽高比。
- SGD momentum 0.9，weight decay 0.0005，masked BCE 的 `valid_mean` reduction。
- 全图 backbone / 新层初始学习率分别为 1e-5 / 0.01，在 500、1,000 步衰减。
- Fisher 阶段 backbone / 投影 / Fisher / 分类头初始学习率为 1e-5 / 1e-4 / 0.001 / 0.001，
  在 3,000、4,000 步衰减，各组均乘 0.1。
- SVM：`C=1`，分别只在 train 特征上拟合；双尺度原始 FV 求均值后 signed sqrt + L2。
- 种子 42；单次神经网络训练，没有多种子重复。

## 核验与产物

训练正常完成，checkpoint 均不是 debug、preparation 或 diagnostic 权重。
全图阶段有 162 个 backbone 浮点 state tensor 相对原始预训练权重发生变化；
Fisher 阶段也有 162 个 backbone tensor 发生变化，投影、Fisher 和分类头参数均发生变化。
周期性诊断记录了各组非零梯度；保存 checkpoint 时检查参数和优化器状态的有限性。
两组 SVM 的 20 类求解器全部收敛。

结果核验检查了完整 val ID、VOC 原始标签、train/val 不重叠、分数有限性，
并从保存的预测重新计算 mAP。配对 bootstrap 只描述验证图像采样的不确定性，
不能替代多种子重新训练。
500 次配对 bootstrap 中，训练后减去训练前的 mAP 差值，其 95% 百分位区间为
[-1.44, -0.06] 个百分点；本次结果不支持额外训练带来提升。

所有原始产物位于 `05_results/dinov3_finetune_20261011/`：

- `whole_image.pt`：全图阶段 1,500 步 checkpoint。
- `fisher_gmm.pt`：全图训练 + GMM 初始化后的模型。
- `fisher_trained.pt`：5,000 步 FisherNet checkpoint，包含优化器和调度器状态。
- `initialized_validation/`、`trained_validation/`：两组 SVM 指标和逐图预测。
- `whole_validation/`、`trained_head_validation/`：分类头指标和逐图预测。
- `experiment_plan.json`、`summary.json`：配置、来源哈希、结果与审计。
- `whole_train.log`、`gmm.log`、`fisher_train.log`、`fisher_diagnostics.jsonl`：训练证据。

便携配置为 `configs/voc2007_dinov3_finetune.yaml`；本机已配置好权重路径的副本为
`configs/voc2007_dinov3_finetune.local.yaml`，本地副本不纳入 Git。
后续训练建议先在 train/val 上调整优化器、各模块学习率或解冻范围，
并继续保留 GMM 初始化模型作为对照；当前结果不能证明继续增加训练步数就会提升。

## 项目导航

完整方案比较见[实验总结](submission_report.md)，所有协议记录见[结果索引](../results/README.md)，
运行命令与本地权重配置见[实现说明](../README.md)。本报告中的实验数值保持为原始已完成结果。

后续已完成 [DINOv2 / DINOv3 端到端与正式 test 对照](backbone_comparison_20261013.md)；本报告仍记录原日期的独立协议。
