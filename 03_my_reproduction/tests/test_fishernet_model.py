import torch

from models.fishernet import FisherNetAlexNet


def test_fishernet_alexnet_forward_and_backward():
    model = FisherNetAlexNet(num_classes=20, descriptor_dim=8, num_components=4)
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (2, 20)
    logits.mean().backward()
    assert model.descriptor_projection.weight.grad is not None
    assert model.fisher.w.grad is not None
