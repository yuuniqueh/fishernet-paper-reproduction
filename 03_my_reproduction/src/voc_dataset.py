"""Minimal PASCAL VOC image-level multi-label dataset reader."""

from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

_NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class ResizeLongestSide:
    """Resize without cropping, preserving the aspect ratio required by FisherNet."""

    def __init__(self, longest_side: int, horizontal_flip: bool = False):
        if longest_side <= 0:
            raise ValueError("longest_side must be positive")
        self.longest_side = longest_side
        self.horizontal_flip = horizontal_flip

    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        scale = self.longest_side / max(width, height)
        resized = TF.resize(image, (max(1, round(height * scale)), max(1, round(width * scale))))
        if self.horizontal_flip:
            resized = TF.hflip(resized)
        return _NORMALIZE(TF.to_tensor(resized))


class RandomFisherNetTransform:
    """Paper augmentation: one of five longest-side scales plus random flip."""

    def __init__(self, image_scales, horizontal_flip: bool = True):
        self.image_scales = tuple(image_scales)
        if not self.image_scales or any(scale <= 0 for scale in self.image_scales):
            raise ValueError("image_scales must contain positive values")
        self.horizontal_flip = horizontal_flip

    def __call__(self, image: Image.Image) -> torch.Tensor:
        transform = ResizeLongestSide(
            random.choice(self.image_scales),
            horizontal_flip=self.horizontal_flip and random.random() < 0.5,
        )
        return transform(image)


def whole_image_finetune_transform():
    """Fixed-size AlexNet input with the paper's horizontal-flip augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


class VOCMultiLabel(Dataset):
    def __init__(
        self,
        root,
        year="2007",
        split="trainval",
        transform=None,
        return_valid_mask=False,
        return_original_size=False,
    ):
        self.base = Path(root) / f"VOC{year}"
        self.transform = transform or transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.return_valid_mask = return_valid_mask
        self.return_original_size = return_original_size
        split_file = self.base / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"missing VOC split file: {split_file}")
        image_ids = [line.strip().split()[0] for line in split_file.read_text().splitlines() if line.strip()]
        if not image_ids or len(set(image_ids)) != len(image_ids):
            raise ValueError(f"VOC split is empty or contains duplicate image IDs: {split_file}")
        missing_images = [
            image_id for image_id in image_ids
            if not (self.base / "JPEGImages" / f"{image_id}.jpg").exists()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"VOC{year} {split} is incomplete: {len(missing_images)} JPEG images are missing "
                f"(for example {missing_images[0]}.jpg). Re-download and extract the full archive."
            )
        tables = {}
        for cls in VOC_CLASSES:
            label_file = self.base / "ImageSets" / "Main" / f"{cls}_{split}.txt"
            rows = label_file.read_text().splitlines()
            tables[cls] = {row.split()[0]: int(row.split()[1]) for row in rows if row.strip()}
        target_rows, valid_rows = [], []
        for image_id in image_ids:
            missing_labels = [cls for cls in VOC_CLASSES if image_id not in tables[cls]]
            if missing_labels:
                raise ValueError(f"missing class labels for {image_id}: {missing_labels}")
            raw_labels = [tables[cls][image_id] for cls in VOC_CLASSES]
            if any(label not in {-1, 0, 1} for label in raw_labels):
                raise ValueError(f"invalid VOC label for {image_id}; expected -1, 0 or 1")
            labels = [float(label == 1) for label in raw_labels]
            # VOC uses 0 for difficult examples: exclude only these from BCE/AP.
            valid = [float(label != 0) for label in raw_labels]
            target_rows.append(labels)
            valid_rows.append(valid)
        # Two contiguous storages, rather than two separately allocated tensors
        # per image. Windows spawn otherwise shares thousands of tiny storages.
        # Row views preserve the historical items/__getitem__ interface exactly.
        self.targets = torch.tensor(target_rows, dtype=torch.float32)
        self.valid_masks = torch.tensor(valid_rows, dtype=torch.float32)
        self.items = [(image_id, self.targets[index], self.valid_masks[index])
                      for index, image_id in enumerate(image_ids)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, target, valid = self.items[index]
        with Image.open(self.base / "JPEGImages" / f"{image_id}.jpg") as source:
            image = source.convert("RGB")
            width, height = image.size
            transformed = self.transform(image)
        if self.return_valid_mask:
            result = (transformed, target, valid, image_id)
        else:
            result = (transformed, target, image_id)
        if self.return_original_size:
            return (*result, torch.tensor([height, width], dtype=torch.int64))
        return result
