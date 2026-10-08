import torch
import pytest

from src.patches import dense_patch_boxes, scale_boxes_between_images, scale_boxes_to_feature_map
from torchvision.ops import roi_pool
from src.patches import half_open_boxes_to_feature_rois


def test_dense_patches_cover_bottom_right_corner():
    boxes = dense_patch_boxes(100, 100, patch_sizes=(64,), step=32)
    assert boxes.shape == (9, 4)
    assert torch.equal(boxes[-1], torch.tensor([36.0, 36.0, 100.0, 100.0]))


def test_scale_boxes_to_feature_map():
    box = torch.tensor([[0.0, 0.0, 224.0, 224.0]])
    scaled = scale_boxes_to_feature_map(box, 224, 224, 13, 13)
    assert torch.allclose(scaled, torch.tensor([[0.0, 0.0, 13.0, 13.0]]))


def test_original_coordinate_boxes_keep_their_count_after_image_resize():
    original_boxes = dense_patch_boxes(375, 500)
    resized_boxes = scale_boxes_between_images(original_boxes, 375, 500, 900, 1200)
    assert original_boxes.shape == resized_boxes.shape == (700, 4)
    assert torch.allclose(resized_boxes[-1], original_boxes[-1] * 2.4)


def test_empty_patches_have_consistent_shape():
    assert dense_patch_boxes(20, 20).shape == (0, 4)
    with pytest.raises(ValueError):
        dense_patch_boxes(100, 100, step=0)
@pytest.mark.parametrize("height,width,fh,fw", [
    (224, 224, 13, 13), (360, 480, 21, 29), (900, 1200, 55, 74),
    (100, 101, 5, 5), (64, 64, 1, 1),
])
def test_half_open_mapping_has_no_artificial_empty_bins(height, width, fh, fw):
    boxes = dense_patch_boxes(height, width)
    rois = half_open_boxes_to_feature_rois(boxes, height, width, fh, fw)
    assert torch.all(rois[:, :2] <= rois[:, 2:])
    assert torch.all(rois[:, [0, 2]] < fw)
    assert torch.all(rois[:, [1, 3]] < fh)
    features = torch.ones(1, 1, fh, fw, requires_grad=True)
    pooled = roi_pool(features, [rois], (6, 6))
    assert torch.all(pooled == 1)
    pooled.sum().backward()
    assert torch.isfinite(features.grad).all()


def test_half_open_full_grid_and_single_cell():
    boxes = torch.tensor([[0., 0., 224., 224.], [223., 223., 224., 224.]])
    mapped = half_open_boxes_to_feature_rois(boxes, 224, 224, 13, 13)
    assert torch.equal(mapped, torch.tensor([[0., 0., 12., 12.], [12., 12., 12., 12.]]))
