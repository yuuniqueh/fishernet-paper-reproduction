"""Whole-image fine-tuning stage that precedes Fisher Layer initialization."""

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet


class WholeImageAlexNet(nn.Module):
    """ImageNet AlexNet adapted to VOC multi-label classification.

    The 256-dimensional descriptor layer is later reused to produce patch
    descriptors for GMM fitting and FisherNet training.
    """

    def __init__(
        self,
        num_classes: int = 20,
        descriptor_dim: int = 256,
        pretrained: bool = True,
        fc_dropout: bool = False,
        new_layer_init_std: float | None = None,
    ):
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        network = alexnet(weights=weights)
        self.backbone = network.features
        self.avgpool = network.avgpool
        self.fc6 = network.classifier[1]
        self.fc7 = network.classifier[4]
        self.relu = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5) if fc_dropout else nn.Identity()
        self.dropout7 = nn.Dropout(p=0.5) if fc_dropout else nn.Identity()
        self.descriptor_projection = nn.Linear(4096, descriptor_dim)
        self.classifier = nn.Linear(descriptor_dim, num_classes)
        if new_layer_init_std is not None:
            for layer in (self.descriptor_projection, self.classifier):
                nn.init.normal_(layer.weight, mean=0.0, std=new_layer_init_std)
                nn.init.zeros_(layer.bias)

    def extract_descriptor(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images)
        x = self.avgpool(x).flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout6(x)
        x = self.relu(self.fc7(x))
        x = self.dropout7(x)
        return self.descriptor_projection(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_descriptor(images))
