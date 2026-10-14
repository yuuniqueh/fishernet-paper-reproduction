"""Dense pretrained backbones shared by whole-image and Fisher stages.

Inputs are already ImageNet-normalized by voc_dataset. ViT images are resized
to the next patch multiple so border pixels are retained. Only spatial tokens
enter the Fisher encoder: CLS and register tokens are explicitly removed.
"""

from contextlib import nullcontext
import importlib
from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.fisher_layer import FisherLayer


MODEL_IDS = {
    "dinov2": "facebook/dinov2-small",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}


class DenseBackbone(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.kind = cfg["backbone"]
        self.backend = cfg.get("backbone_backend", "transformers")
        self.frozen = cfg.get("freeze_backbone", False)
        if self.kind == "dinov3" and self.backend == "official":
            repository = Path(cfg["backbone_repo"]).resolve()
            if not (repository / "hubconf.py").is_file():
                raise FileNotFoundError(f"DINOv3 source repository missing: {repository}")
            architecture = cfg.get("backbone_arch", "dinov3_vits16")
            if architecture not in {"dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16"}:
                raise ValueError("supported local DINOv3 ViTs: S/16, S+/16, B/16, L/16")
            # Construct offline, then load the file directly. No torch-hub
            # weight download or copy into the user's global cache is needed.
            sys.path.insert(0, str(repository))
            try:
                constructors = importlib.import_module("dinov3.hub.backbones")
                if not Path(constructors.__file__).resolve().is_relative_to(repository):
                    raise ValueError("another DINOv3 source tree is already imported; start a new Python process")
                # Import only the backbone, avoiding unrelated segmentation
                # dependencies pulled in by the repository's hubconf.py.
                self.network = getattr(constructors, architecture)(pretrained=False)
            finally:
                sys.path.pop(0)
            if pretrained:
                weights = Path(cfg["backbone_weights"]).resolve()
                state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
                self.network.load_state_dict(state, strict=True)
                del state
            self.output_dim, self.patch_size = self.network.embed_dim, self.network.patch_size
            cfg["backbone_config"] = dict(architecture=architecture, hidden_size=self.output_dim,
                                           patch_size=self.patch_size, depth=len(self.network.blocks))
        elif self.kind == "resnet50":
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.network = nn.Sequential(*list(network.children())[:-2])
            self.output_dim, self.patch_size = 2048, 32
        elif self.kind in MODEL_IDS:
            try:
                from transformers import AutoModel, Dinov2Config, Dinov2Model
                from transformers import DINOv3ViTConfig, DINOv3ViTModel
            except ImportError as error:
                raise ImportError("DINO backbones require requirements-backbones.txt") from error
            source = cfg.get("backbone_model_name", MODEL_IDS[self.kind])
            if pretrained:
                self.network = AutoModel.from_pretrained(
                    source, cache_dir=cfg.get("backbone_cache_dir"),
                    local_files_only=cfg.get("backbone_local_files_only", False),
                )
            else:
                # Save the actual architecture in checkpoints, avoiding network
                # access when restoring trained models, including custom sizes.
                settings = cfg.get("backbone_config")
                if settings is None and source != MODEL_IDS[self.kind]:
                    raise ValueError("custom DINO checkpoint requires backbone_config to restore offline")
                if self.kind == "dinov2":
                    settings = settings or dict(hidden_size=384, num_attention_heads=6)
                    self.network = Dinov2Model(Dinov2Config(**settings))
                else:
                    settings = settings or dict(num_register_tokens=4)
                    self.network = DINOv3ViTModel(DINOv3ViTConfig(**settings))
            if self.network.config.model_type != {"dinov2": "dinov2", "dinov3": "dinov3_vit"}[self.kind]:
                raise ValueError("backbone_model_name does not match configured DINO family")
            cfg["backbone_config"] = self.network.config.to_dict()
            self.output_dim = self.network.config.hidden_size
            self.patch_size = self.network.config.patch_size
        else:
            raise ValueError(f"unknown backbone: {self.kind}")
        if self.frozen:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode=True):
        # Frozen BatchNorm and DINO stochastic layers must stay in eval mode.
        return super().train(False if self.frozen else mode)

    def descriptor_count(self, height, width):
        return math.ceil(height / self.patch_size) * math.ceil(width / self.patch_size)

    def _features(self, images):
        if self.kind == "resnet50":
            patches = self.network(images).flatten(2).transpose(1, 2)
            return patches, patches.mean(1)
        height, width = images.shape[-2:]
        size = tuple(math.ceil(n / self.patch_size) * self.patch_size for n in (height, width))
        if size != (height, width):
            images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        if self.backend == "official":
            features = self.network.forward_features(images)
            patches = features["x_norm_patchtokens"]
            if patches.shape[1] != self.descriptor_count(height, width):
                raise RuntimeError("unexpected DINOv3 spatial token count")
            return patches, features["x_norm_clstoken"]
        tokens = self.network(pixel_values=images).last_hidden_state
        start = 1 + getattr(self.network.config, "num_register_tokens", 0)
        patches = tokens[:, start:]
        if patches.shape[1] != self.descriptor_count(height, width):
            raise RuntimeError("unexpected spatial token count; check model architecture")
        return patches, tokens[:, 0]

    def features(self, images):
        with torch.no_grad() if self.frozen else nullcontext():
            return self._features(images)

    def forward(self, images):
        return self.features(images)[0]


class WholeImageModern(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.backbone = DenseBackbone(cfg, pretrained)
        self.descriptor_projection = nn.Linear(self.backbone.output_dim, cfg["descriptor_dim"])
        self.classifier = nn.Linear(cfg["descriptor_dim"], cfg["num_classes"])

    def extract_descriptor(self, images):
        # The same projection maps local tokens in the Fisher stage.
        return self.descriptor_projection(self.backbone(images).mean(1))

    def forward(self, images):
        return self.classifier(self.extract_descriptor(images))


class FisherNetModern(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.backbone = DenseBackbone(cfg, pretrained)
        self.descriptor_projection = nn.Linear(self.backbone.output_dim, cfg["descriptor_dim"])
        self.fisher = FisherLayer(cfg["descriptor_dim"], cfg["num_components"])
        self.classifier = nn.Linear(2 * cfg["num_components"] * cfg["descriptor_dim"], cfg["num_classes"])
        self.patch_coordinate_mode = "resized"
        self.max_patches = cfg.get("max_patches")
        if self.max_patches is not None and self.max_patches <= 0:
            raise ValueError("max_patches must be positive or None")

    def descriptor_count(self, height, width):
        count = self.backbone.descriptor_count(height, width)
        return min(count, self.max_patches) if self.max_patches else count

    def extract_descriptors(self, images, original_sizes=None, patch_indices=None):
        patches = self.backbone(images)
        if self.max_patches:
            # Spread a bounded subset across the image rather than the top edge.
            indices = torch.linspace(0, patches.shape[1] - 1,
                                     min(self.max_patches, patches.shape[1]), device=images.device).long()
            patches = patches.index_select(1, indices)
        if patch_indices is not None:
            if (patch_indices.ndim != 1 or patch_indices.dtype != torch.int64
                    or patch_indices.numel() == 0 or patch_indices.min() < 0
                    or patch_indices.max() >= patches.shape[1]):
                raise ValueError("patch_indices must be a non-empty, in-range int64 vector")
            patches = patches.index_select(1, patch_indices.to(images.device))
        return self.descriptor_projection(patches)

    def extract_fisher_vectors(self, images, original_sizes=None):
        return self.fisher(self.extract_descriptors(images, original_sizes))

    def forward(self, images, original_sizes=None):
        return self.classifier(self.extract_fisher_vectors(images, original_sizes))
