# DINOv2 / DINOv3 冻结特征验证实验

2026-10-11 在本机 RTX 4060 Laptop（8GB）运行完整 VOC2007 train→val 实验。
训练和验证分别为 2,501 / 2,510 张图像，无 ID 重叠。验证集标签仅用于最终 AP，
PCA、GMM 和 SVM 全部只拟合训练集。没有使用 test 选择 backbone 或调参。

| 模型 | 特征 / 分类方式 | val mAP |
| --- | --- | ---: |
| 既有 AlexNet | 原训练流程、五尺度 Fisher / SVM | 72.29% |
| DINOv2-small | 冻结 CLS + 空间 token 均值 / SVM | **89.37%** |
| DINOv2-small | 冻结空间 token / PCA / Fisher / SVM | 88.00% |
| DINOv3 ViT-S/16 | 冻结 CLS + 空间 token 均值 / SVM | 88.68% |
| DINOv3 ViT-S/16 | 冻结空间 token / PCA / Fisher / SVM | 85.27% |

两项 DINO 实验的 20 类 SVM 均收敛，GMM 均收敛。各类别 AP、求解器迭代数、权重标识及哈希
保存在 `results/backbone_trials_20261011.json`。

## 协议与参数

- 全图缩放为 224×224，保留整图内容，不做中心裁剪；ImageNet mean/std 归一化。
- DINOv2-small 使用公开 Hugging Face 权重，384 维、14 像素 patch，256 个空间 token。
- DINOv3 使用用户已有的官方 `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`，
  源码位于 `D:/enter_project/third_party/dinov3-main`，384 维、16 像素 patch，196 个空间 token。
- 两个 backbone 均冻结并处于 eval 模式，无 VOC 微调。局部 Fisher 不包含 CLS 或 register token。
- 磁盘缓存局部 token 为 float16，随后以 float32 拟合及编码；全局特征缓存为 float32。
- 从全部训练图像均匀分配采样预算，取得 50,000 个局部描述子；随机种子 42。
- PCA：64 维，whitening，randomized SVD；GMM：16 个对角协方差分量，`reg_covar=1e-4`，最多 300 次迭代。
- Fisher 使用项目现有可微 FisherLayer 的参数化方式，以 GMM 均值/方差初始化，
  编码后 signed square root + L2；全局特征仅 L2。
- 所有线性 SVM 固定 `C=1`，最多 10,000 次迭代，排除 difficult 标签；VOC2007 11-point AP。
- Python 3.11、PyTorch 2.8.0+cu128、torchvision 0.23.0、Transformers 4.57.6（仅 DINOv2 后端使用）。

## 结论的范围

更强预训练特征在这次完整独立验证中明显高于旧 AlexNet 验证结果。
但预训练数据、缩放方式、局部描述子、PCA 和冻结训练方式均有变化，无法把差值全部归因于 backbone。
DINOv2-small 在当前协议下比 DINOv3 ViT-S/16 高，全局特征 SVM 也比 Fisher 路线高。
这说明应同时评估聚合方式，不能假定更新的 backbone 或增加 Fisher 一定提升效果。

本次只验证冻结特征路线，未宣称新骨干的完整端到端微调已经完成，也未产生新的 test 指标。
后续已完成 DINOv3 [短程训练](dinov3_finetune_20261011.md)和独立的
[全图 9,000 + Fisher 40,000 步实验](dinov3_40000_20261011.md)。后者最终 val mAP 为 87.24%，
所评估检查点中验证最佳为 87.29%；本文件保留冻结协议及其历史数值，不与微调协议合并排名。
完整方案概览见[实验总结](submission_report.md)和[结果索引](../results/README.md)。
本报告之后已完成 DINOv2 / DINOv3 的 trainval/test，对照见下方新实验链接；此前冻结指标保持原样。

## 接入与验证

模型工厂已覆盖 `train_whole_image.py`、`initialize_fisher.py`、`train.py`、
`evaluate_whole_image.py` 和 `evaluate.py`。新 checkpoint 记录骨干架构；恢复已有模型时不下载权重。
兼容性检查拒绝错误骨干、后端或架构。已有 AlexNet 配置及 checkpoint 格式继续保留。

完整测试覆盖空间 token 排除、非整除图像边缘、冻结 BatchNorm、投影迁移、
Fisher 反向传播、离线恢复、严格加载本地权重及 PCA 数值一致性。
真实本地 DINOv3 四阶段流程另外使用独立的小规模 debug 配置检查，产物仅用于可运行性验证，
不计入上表的精度结果。

原始试验目录：

- `05_results/backbone_dinov2_validation_20261011/`
- `05_results/backbone_dinov3_validation_20261011/`
- `05_results/backbone_pipeline_smoke_20261011/`

复现命令和本地权重配置见[实现说明](../README.md)。

后续已完成 [DINOv2 / DINOv3 端到端与正式 test 对照](backbone_comparison_20261013.md)；本报告仍记录原日期的独立协议。
