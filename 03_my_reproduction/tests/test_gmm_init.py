import numpy as np
import torch
import pytest

from src.fisher_layer import FisherLayer
from src.gmm_init import initialize_fisher_from_gmm


def test_gmm_initializes_fisher_parameters():
    layer = FisherLayer(descriptor_dim=2, num_components=2)
    means = np.array([[2.0, -4.0], [1.0, 3.0]])
    variances = np.array([[4.0, 9.0], [1.0, 16.0]])
    initialize_fisher_from_gmm(layer, means, variances)
    assert torch.allclose(layer.b, torch.tensor([[-2.0, 4.0], [-1.0, -3.0]]))
    assert torch.allclose(layer.w, torch.tensor([[0.5, 1 / 3], [1.0, 0.25]]))


@pytest.mark.parametrize("bad_variance", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_gmm_parameters_are_rejected(bad_variance):
    layer = FisherLayer(descriptor_dim=2, num_components=2)
    with pytest.raises(ValueError):
        initialize_fisher_from_gmm(layer, np.zeros((2, 2)), np.full((2, 2), bad_variance))
