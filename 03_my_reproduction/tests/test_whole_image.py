import pytest
import torch

from models.whole_image import WholeImageAlexNet


def test_whole_image_model_forward_and_backward():
    model = WholeImageAlexNet(num_classes=20, descriptor_dim=8, pretrained=False)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 20)
    logits.mean().backward()
    assert model.descriptor_projection.weight.grad is not None


def test_paper_style_fc_dropout_and_new_layer_initialization_are_configurable():
    torch.manual_seed(7)
    model = WholeImageAlexNet(
        num_classes=20,
        descriptor_dim=8,
        pretrained=False,
        fc_dropout=True,
        new_layer_init_std=0.01,
    )
    assert isinstance(model.dropout6, torch.nn.Dropout)
    assert isinstance(model.dropout7, torch.nn.Dropout)
    assert model.descriptor_projection.weight.std().item() == pytest.approx(0.01, abs=0.002)
    assert torch.equal(model.classifier.bias, torch.zeros_like(model.classifier.bias))
