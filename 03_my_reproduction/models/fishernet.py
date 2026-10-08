"""First runnable PyTorch FisherNet model.

This stage uses dense convolutional features as local descriptors.  It preserves
the key end-to-end route (CNN -> local descriptors -> Fisher Layer -> logits).
The separate fishernet_spp module implements the multi-scale RoI/SPP route;
exact equivalence to the authors' unpublished pooling remains unverified.
"""

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

from src.fisher_layer import FisherLayer


class FisherNetAlexNet(nn.Module):
    """AlexNet backbone plus a differentiable Fisher Layer for multi-label VOC."""

    def __init__(
        self,
        num_classes: int = 20,
        descriptor_dim: int = 256,
        num_components: int = 32,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = alexnet(weights=weights)
        # Keep convolution layers up to conv5/ReLU; local spatial cells are patches.
        self.backbone = backbone.features[:12]
        self.descriptor_projection = nn.Conv2d(256, descriptor_dim, kernel_size=1)
        self.fisher = FisherLayer(descriptor_dim, num_components)
        self.classifier = nn.Linear(2 * num_components * descriptor_dim, num_classes)

    def extract_descriptors(self, images: torch.Tensor) -> torch.Tensor:
        """Return local descriptors with shape [batch, patches, descriptor_dim]."""
        feature_map = self.backbone(images)
        descriptors = self.descriptor_projection(feature_map)
        return descriptors.flatten(start_dim=2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        descriptors = self.extract_descriptors(images)
        representation = self.fisher(descriptors)
        return self.classifier(representation)
