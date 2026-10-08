from pathlib import Path

from PIL import Image
import pytest
import torch

from src.voc_dataset import VOCMultiLabel, VOC_CLASSES, ResizeLongestSide, RandomFisherNetTransform


def make_fixture(tmp_path, missing_class=None):
    base = tmp_path / "VOC2007"
    labels = base / "ImageSets/Main"
    labels.mkdir(parents=True)
    (base / "JPEGImages").mkdir()
    Image.new("RGB", (100, 80)).save(base / "JPEGImages/000001.jpg")
    (labels / "trainval.txt").write_text("000001\n", encoding="utf-8")
    for index, name in enumerate(VOC_CLASSES):
        label = [1, -1, 0][index % 3]
        text = "" if name == missing_class else f"000001 {label}\n"
        (labels / f"{name}_trainval.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_voc_difficult_mask_and_original_size(tmp_path):
    root = make_fixture(tmp_path)
    image, targets, valid, image_id, size = VOCMultiLabel(
        root, return_valid_mask=True, return_original_size=True)[0]
    assert image.shape == (3, 224, 224) and image_id == "000001"
    assert size.tolist() == [80, 100]
    assert targets[:3].tolist() == [1.0, 0.0, 0.0]
    assert valid[:3].tolist() == [1.0, 1.0, 0.0]


def test_missing_class_label_does_not_silently_become_difficult(tmp_path):
    with pytest.raises(ValueError, match="missing class labels"):
        VOCMultiLabel(make_fixture(tmp_path, missing_class="cat"))


def test_labels_share_two_contiguous_storages_with_identical_values(tmp_path):
    root = make_fixture(tmp_path)
    base = root / "VOC2007"
    Image.new("RGB", (100, 80)).save(base / "JPEGImages/000002.jpg")
    (base / "ImageSets/Main/trainval.txt").write_text("000001\n000002\n", encoding="utf-8")
    for index, name in enumerate(VOC_CLASSES):
        label = [1, -1, 0][index % 3]
        (base / f"ImageSets/Main/{name}_trainval.txt").write_text(
            f"000001 {label}\n000002 {label}\n", encoding="utf-8")
    dataset = VOCMultiLabel(root, return_valid_mask=True)
    assert len({target.untyped_storage().data_ptr() for _, target, _ in dataset.items}) == 1
    assert len({valid.untyped_storage().data_ptr() for _, _, valid in dataset.items}) == 1
    assert torch.equal(dataset[0][1], dataset[1][1])
    assert torch.equal(dataset[0][2], dataset[1][2])
    assert dataset.targets.is_contiguous() and dataset.valid_masks.is_contiguous()


def test_invalid_transforms_rejected():
    with pytest.raises(ValueError):
        ResizeLongestSide(0)
    with pytest.raises(ValueError):
        RandomFisherNetTransform([])
