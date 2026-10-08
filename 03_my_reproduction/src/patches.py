"""Dense multi-scale patch generation used by the paper's SPP pipeline."""

import torch


def dense_patch_boxes(
    height: int,
    width: int,
    patch_sizes=(64, 96, 128, 160, 192, 224, 256),
    step: int = 32,
    device=None,
) -> torch.Tensor:
    """Generate half-open image-edge boxes [x1, y1, x2, y2).

    Right/bottom endpoints may equal width/height, not width-1/height-1.
    """
    if height <= 0 or width <= 0 or step <= 0 or any(size <= 0 for size in patch_sizes):
        raise ValueError("image dimensions, patch sizes and step must be positive")
    boxes = []
    for size in patch_sizes:
        if size > height or size > width:
            continue
        xs = list(range(0, width - size + 1, step))
        ys = list(range(0, height - size + 1, step))
        if xs[-1] != width - size:
            xs.append(width - size)
        if ys[-1] != height - size:
            ys.append(height - size)
        boxes.extend((x, y, x + size, y + size) for y in ys for x in xs)
    return torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)


def scale_boxes_to_feature_map(
    boxes: torch.Tensor,
    image_height: int,
    image_width: int,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Convert image-coordinate boxes to feature-map coordinates for RoI pooling."""
    scaled = boxes.clone()
    scaled[:, [0, 2]] *= feature_width / image_width
    scaled[:, [1, 3]] *= feature_height / image_height
    return scaled


def scale_boxes_between_images(
    boxes: torch.Tensor,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Map boxes from one image resolution to another without changing count."""
    scaled = boxes.clone()
    scaled[:, [0, 2]] *= target_width / source_width
    scaled[:, [1, 3]] *= target_height / source_height
    return scaled


def half_open_boxes_to_feature_rois(
    boxes: torch.Tensor,
    image_height: int,
    image_width: int,
    feature_height: int,
    feature_width: int,
) -> torch.Tensor:
    """Map half-open image edges to bounded *inclusive* RoI-pool indices.

    Select every intersected feature cell: floor(start), ceil(end)-1, then
    clamp to the valid feature grid. TorchVision roi_pool includes both end
    indices; feeding a half-open end directly creates out-of-grid bins.
    This defines a consistent ratio mapping, not the author's unknown mapping.
    The historical conversion above remains unchanged for old checkpoints.
    """
    if min(image_height, image_width, feature_height, feature_width) <= 0:
        raise ValueError("image and feature dimensions must be positive")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    scaled = scale_boxes_to_feature_map(
        boxes, image_height, image_width, feature_height, feature_width
    )
    starts = scaled[:, :2].floor()
    ends = scaled[:, 2:].ceil() - 1
    limits = scaled.new_tensor([feature_width - 1, feature_height - 1])
    starts = torch.minimum(starts.clamp_min(0), limits)
    ends = torch.minimum(ends.clamp_min(0), limits)
    # Even a sub-cell legal patch selects one feature cell, never an empty RoI.
    return torch.cat((starts, torch.maximum(starts, ends)), dim=1)
