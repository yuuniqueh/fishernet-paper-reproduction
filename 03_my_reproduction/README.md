# 运行说明

以下命令在 `03_my_reproduction/` 下执行。每次实验使用新的输出目录。

## 环境与数据

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
```

DINOv2 的 Transformers 后端另需安装：

```powershell
python -m pip install -r requirements-backbones.txt
```

VOC2007 目录为 `../04_dataset/VOC/VOCdevkit/VOC2007/`，其中包含 `JPEGImages/`、`Annotations/` 和 `ImageSets/Main/`。

| 划分 | 图片数 | 用途 |
| --- | ---: | --- |
| train | 2,501 | 验证实验训练 |
| val | 2,510 | 验证实验评估 |
| trainval | 5,011 | 正式测试模型训练 |
| test | 4,952 | 正式测试评估 |

网络、GMM 和 SVM 只拟合对应训练集。train/val 与 trainval/test 分别训练模型，结果分别报告。

## 模型配置

| 配置 | 用途 |
| --- | --- |
| `voc2007_alexnet.yaml` | AlexNet trainval/test |
| `voc2007_dinov2_40000.yaml` | DINOv2 train/val |
| `voc2007_dinov3_40000.yaml` | DINOv3 train/val |
| `voc2007_dinov2_40000_trainval.yaml` | DINOv2 trainval/test |
| `voc2007_dinov3_40000_trainval.yaml` | DINOv3 trainval/test |

完整预算为全图 9,000 步、FisherNet 40,000 步。DINO 使用 256 维描述子、32 分量 GMM、224/336 双尺度和线性 SVM（C=1）。

### DINOv3 本地权重

复制配置为本地副本，并加入官方源码与权重路径：

```powershell
Copy-Item configs/voc2007_dinov3_40000_trainval.yaml configs/voc2007_dinov3_40000_trainval.local.yaml
```

```yaml
backbone_backend: official
backbone_repo: D:/enter_project/third_party/dinov3-main
backbone_weights: D:/enter_project/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
backbone_arch: dinov3_vits16
```

将路径替换为本机位置；官方源码、S/16 架构和权重须匹配。`.local.yaml` 不提交到 Git。

DINOv2 可使用已缓存的 `facebook/dinov2-small`，或将 `backbone_model_name` 改为本地 Hugging Face 模型目录，并设置 `backbone_local_files_only: true`。

## 训练与评估

以下以 DINOv2 的 trainval/test 为例。运行 AlexNet 时替换配置；运行 DINOv3 时使用上面的本地配置副本，并更换输出目录。

```powershell
python train_whole_image.py --config configs/voc2007_dinov2_40000_trainval.yaml --output ../05_results/dinov2_rerun/whole_image.pt
python initialize_fisher.py --config configs/voc2007_dinov2_40000_trainval.yaml --whole-image-checkpoint ../05_results/dinov2_rerun/whole_image.pt --output ../05_results/dinov2_rerun/fisher_gmm.pt
python train.py --config configs/voc2007_dinov2_40000_trainval.yaml --init-checkpoint ../05_results/dinov2_rerun/fisher_gmm.pt --output ../05_results/dinov2_rerun/fisher_trained.pt
python evaluate.py --config configs/voc2007_dinov2_40000_trainval.yaml --checkpoint ../05_results/dinov2_rerun/fisher_trained.pt --protocol svm --split test --fit-split trainval --output-dir ../05_results/dinov2_rerun/evaluation_final
```

评估初始化模型时，将 checkpoint 换为 `fisher_gmm.pt` 并使用新的输出目录。运行 train/val 时使用对应配置，并将评估参数设为 `--split val --fit-split train`。

### 一次运行完整对照

`run_backbone_comparison.py` 依次运行 DINOv2 train/val、DINOv2 trainval/test、DINOv3 trainval/test，保存固定检查点并核验预测。DINOv2 需已缓存 revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`。

```powershell
python run_backbone_comparison.py --output-dir ../05_results/comparison_rerun --dinov3-repo D:/enter_project/third_party/dinov3-main --dinov3-weights D:/enter_project/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

初始化模型的辅助评估需按上述评估入口另行执行。冻结特征对照使用 `benchmark_backbone.py`。

结果见[实验记录](docs/backbone_comparison_20261013.md)和[结果索引](results/README.md)。
