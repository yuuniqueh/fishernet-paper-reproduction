"""One model selection path for training, initialization and evaluation."""

from models.fishernet import FisherNetAlexNet
from models.fishernet_spp import FisherNetSPPAlexNet
from models.whole_image import WholeImageAlexNet
from models.modern_backbone import FisherNetModern, WholeImageModern


def build_model(cfg, stage="fisher", pretrained=False):
    if stage not in {"whole_image", "fisher"}:
        raise ValueError(f"unknown model stage: {stage}")
    if cfg["backbone"] != "alexnet":
        if cfg.get("model_variant", "dense") != "dense" or cfg.get("patch_coordinate_mode", "resized") != "resized":
            raise ValueError("modern backbones use dense spatial descriptors in resized coordinates")
        return (WholeImageModern if stage == "whole_image" else FisherNetModern)(cfg, pretrained)
    common = dict(num_classes=cfg["num_classes"], descriptor_dim=cfg["descriptor_dim"])
    if stage == "whole_image":
        return WholeImageAlexNet(**common, pretrained=pretrained,
                                 fc_dropout=cfg.get("fc_dropout", False),
                                 new_layer_init_std=cfg.get("new_layer_init_std"))
    common.update(num_components=cfg["num_components"], pretrained_backbone=pretrained)
    variant = cfg.get("model_variant", "dense")
    if variant == "dense":
        return FisherNetAlexNet(**common)
    if variant != "spp":
        raise ValueError(f"unknown model_variant: {variant}")
    return FisherNetSPPAlexNet(
        **common, patch_sizes=tuple(cfg["patch_scales"]), patch_step=cfg["patch_step"],
        max_patches=cfg.get("max_patches"),
        patch_coordinate_mode=cfg.get("patch_coordinate_mode", "resized"),
        roi_coordinate_mapping=cfg.get("roi_coordinate_mapping", "legacy_ratio"),
        fc_dropout=cfg.get("fc_dropout", False), new_layer_init_std=cfg.get("new_layer_init_std"),
    )
