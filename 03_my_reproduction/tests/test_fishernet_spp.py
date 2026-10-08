import torch
import pytest
import models.fishernet_spp as spp_module

from models.fishernet_spp import FisherNetSPPAlexNet


def test_spp_fishernet_forward_and_backward():
    model = FisherNetSPPAlexNet(
        num_classes=20,
        descriptor_dim=8,
        num_components=4,
        patch_sizes=(64, 96),
        patch_step=64,
        max_patches=3,
    )
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 20)
    logits.mean().backward()
    assert model.fc6.weight.grad is not None
    assert model.fisher.w.grad is not None


def test_stride_mapping_keeps_image_coordinate_rois():
    model = FisherNetSPPAlexNet(
        descriptor_dim=8,
        num_components=4,
        patch_sizes=(64,),
        patch_step=64,
        max_patches=1,
        roi_coordinate_mapping="backbone_stride",
    )
    assert model.roi_spatial_scale == 1 / 16
    assert model.roi_coordinate_mapping == "backbone_stride"


@pytest.mark.parametrize("mapping, expected_scale, expected_end", [
    ("legacy_ratio", 1.0, 64 * 13 / 224), ("backbone_stride", 1 / 16, 64.0),
    ("half_open_ratio", 1.0, 3.0)])
def test_actual_roi_pool_call_uses_selected_mapping(monkeypatch, mapping, expected_scale, expected_end):
    captured = {}
    actual_pool = spp_module.roi_pool
    def record_pool(features, boxes, output_size, spatial_scale):
        captured["boxes"] = boxes[0].detach().clone()
        captured["spatial_scale"] = spatial_scale
        return actual_pool(features, boxes, output_size, spatial_scale)
    monkeypatch.setattr(spp_module, "roi_pool", record_pool)
    model = FisherNetSPPAlexNet(descriptor_dim=8, num_components=2, patch_sizes=(64,),
                               max_patches=1, roi_coordinate_mapping=mapping)
    with torch.inference_mode():
        model(torch.randn(1, 3, 224, 224))
    assert captured["spatial_scale"] == expected_scale
    assert captured["boxes"][0, 2].item() == pytest.approx(expected_end)


def test_early_patch_sampling_matches_selected_full_descriptors():
    model = FisherNetSPPAlexNet(descriptor_dim=8, num_components=2, patch_sizes=(64,), max_patches=5)
    model.eval()
    image = torch.randn(1, 3, 224, 224)
    indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    with torch.inference_mode():
        full = model.extract_descriptors(image)
        sampled = model.extract_descriptors(image, patch_indices=indices)
    assert torch.allclose(sampled, full[:, indices], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="patch_indices"):
        model.extract_descriptors(image, patch_indices=torch.tensor([7]))
