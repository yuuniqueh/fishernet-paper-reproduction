"""SPP-based FisherNet stage, structurally close to the original paper."""

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.ops import roi_pool

from src.fisher_layer import FisherLayer
from src.patches import (dense_patch_boxes, scale_boxes_between_images,
                         scale_boxes_to_feature_map, half_open_boxes_to_feature_rois)


class FisherNetSPPAlexNet(nn.Module):
    """AlexNet -> multi-scale RoI/SPP -> patch descriptors -> Fisher Layer."""

    def __init__(
        self,
        num_classes: int = 20,
        descriptor_dim: int = 256,
        num_components: int = 32,
        patch_sizes=(64, 96, 128, 160, 192, 224, 256),
        patch_step: int = 32,
        max_patches: int | None = None,
        patch_coordinate_mode: str = "resized",
        roi_coordinate_mapping: str = "legacy_ratio",
        pretrained_backbone: bool = False,
        fc_dropout: bool = False,
        new_layer_init_std: float | None = None,
    ):
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        network = alexnet(weights=weights)
        self.backbone = network.features[:12]
        self.fc6 = network.classifier[1]
        self.fc7 = network.classifier[4]
        self.relu = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5) if fc_dropout else nn.Identity()
        self.dropout7 = nn.Dropout(p=0.5) if fc_dropout else nn.Identity()
        self.descriptor_projection = nn.Linear(4096, descriptor_dim)
        self.fisher = FisherLayer(descriptor_dim, num_components)
        self.classifier = nn.Linear(2 * num_components * descriptor_dim, num_classes)
        if new_layer_init_std is not None:
            for layer in (self.descriptor_projection, self.classifier):
                nn.init.normal_(layer.weight, mean=0.0, std=new_layer_init_std)
                nn.init.zeros_(layer.bias)
        self.patch_sizes = patch_sizes
        self.patch_step = patch_step
        self.max_patches = max_patches
        if patch_coordinate_mode not in {"resized", "original"}:
            raise ValueError("patch_coordinate_mode must be 'resized' or 'original'")
        self.patch_coordinate_mode = patch_coordinate_mode
        if roi_coordinate_mapping not in {"legacy_ratio", "backbone_stride", "half_open_ratio"}:
            raise ValueError(
                "roi_coordinate_mapping must be legacy_ratio, backbone_stride or half_open_ratio"
            )
        self.roi_coordinate_mapping = roi_coordinate_mapping
        # AlexNet conv5 (with the final max-pooling layer removed) has an
        # effective spatial stride of 16 pixels. This is a mapping hypothesis,
        # not a verified match to the authors' unpublished SPP implementation.
        self.roi_spatial_scale = 1.0 / 16.0
        if max_patches is not None and max_patches <= 0:
            raise ValueError("max_patches must be positive or None")

    def _image_boxes(
        self,
        image_height: int,
        image_width: int,
        original_sizes: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate the paper patches in resized or original-image coordinates."""
        if self.patch_coordinate_mode == "resized":
            boxes = dense_patch_boxes(
                image_height, image_width, self.patch_sizes, self.patch_step, device
            )
        else:
            if original_sizes is None:
                raise ValueError("original_sizes is required for original-coordinate patches")
            if original_sizes.shape[0] != 1:
                raise ValueError("original-coordinate patches require a batch size of one")
            original_height, original_width = (int(value) for value in original_sizes[0].tolist())
            boxes = dense_patch_boxes(
                original_height, original_width, self.patch_sizes, self.patch_step, device
            )
            boxes = scale_boxes_between_images(
                boxes, original_height, original_width, image_height, image_width
            )
        if self.max_patches is not None:
            boxes = boxes[: self.max_patches]
        return boxes

    def extract_descriptors(
        self, images: torch.Tensor, original_sizes: torch.Tensor | None = None,
        patch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate one descriptor for every dense multi-scale patch."""
        batch_size, _, image_height, image_width = images.shape
        feature_map = self.backbone(images)
        _, _, feature_height, feature_width = feature_map.shape
        image_boxes = self._image_boxes(
            image_height, image_width, original_sizes, images.device
        )
        if image_boxes.shape[0] == 0:
            raise ValueError("no valid patches: image is smaller than every configured patch size")
        if patch_indices is not None:
            if (patch_indices.ndim != 1 or patch_indices.dtype != torch.int64
                    or patch_indices.numel() == 0
                    or patch_indices.min().item() < 0
                    or patch_indices.max().item() >= len(image_boxes)):
                raise ValueError("patch_indices must be a non-empty, in-range int64 vector")
            image_boxes = image_boxes.index_select(0, patch_indices.to(image_boxes.device))
        if self.roi_coordinate_mapping in {"legacy_ratio", "half_open_ratio"}:
            mapping = (half_open_boxes_to_feature_rois
                       if self.roi_coordinate_mapping == "half_open_ratio"
                       else scale_boxes_to_feature_map)
            roi_boxes = mapping(
                image_boxes, image_height, image_width, feature_height, feature_width
            )
            spatial_scale = 1.0
        else:
            roi_boxes = image_boxes
            spatial_scale = self.roi_spatial_scale
        pooled = roi_pool(
            feature_map,
            [roi_boxes for _ in range(batch_size)],
            output_size=(6, 6),
            spatial_scale=spatial_scale,
        )
        patch_features = self.relu(self.fc6(pooled.flatten(start_dim=1)))
        patch_features = self.dropout6(patch_features)
        patch_features = self.relu(self.fc7(patch_features))
        patch_features = self.dropout7(patch_features)
        descriptors = self.descriptor_projection(patch_features)
        return descriptors.reshape(batch_size, image_boxes.shape[0], -1)

    def forward(self, images: torch.Tensor, original_sizes: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.extract_fisher_vectors(images, original_sizes))

    def extract_fisher_vectors(
        self, images: torch.Tensor, original_sizes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return image-level Fisher vectors before the final classifier."""
        return self.fisher(self.extract_descriptors(images, original_sizes))
