import torch
import math
import pytest
from src.fisher_layer import FisherLayer


def test_fisher_layer_shape_and_grad():
    layer = FisherLayer(descriptor_dim=4, num_components=3)
    x = torch.randn(2, 5, 4, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 24)
    y.square().mean().backward()
    assert x.grad is not None
    assert layer.w.grad is not None
    assert layer.b.grad is not None


def test_fisher_values_match_independent_equations_and_gradcheck():
    layer = FisherLayer(descriptor_dim=2, num_components=2).double()
    with torch.no_grad():
        layer.w.copy_(torch.tensor([[0.7, 1.2], [1.5, 0.6]]))
        layer.b.copy_(torch.tensor([[0.2, -0.1], [-0.4, 0.3]]))
    x = torch.tensor([[[0.2, -0.3], [0.5, 0.1]]], dtype=torch.double, requires_grad=True)
    mean_terms, variance_terms = [], []
    for patch in x[0]:
        z = layer.w * (patch + layer.b)
        likelihood = torch.exp(-0.5 * z.square().sum(dim=1))
        gamma = likelihood / likelihood.sum()
        mean_terms.append(gamma[:, None] * z)
        variance_terms.append(gamma[:, None] * (z.square() - 1) / math.sqrt(2))
    reference = torch.cat([torch.stack(mean_terms).mean(0).flatten(),
                           torch.stack(variance_terms).mean(0).flatten()])[None]
    assert torch.allclose(layer(x), reference, atol=1e-12)
    # Include all learnable inputs in finite-difference gradient verification.
    from torch.func import functional_call
    def forward(x_value, w_value, b_value):
        return functional_call(layer, {"w": w_value, "b": b_value}, (x_value,))
    assert torch.autograd.gradcheck(forward, (x, layer.w, layer.b), eps=1e-6, atol=1e-5)


def test_fisher_rejects_empty_patch_set():
    with pytest.raises(ValueError, match="non-empty"):
        FisherLayer(4, 3)(torch.empty(1, 0, 4))
