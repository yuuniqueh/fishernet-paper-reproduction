from PIL import Image

from src.voc_dataset import ResizeLongestSide


def test_resize_longest_side_preserves_aspect_ratio():
    image = Image.new("RGB", (400, 200))
    tensor = ResizeLongestSide(480)(image)
    assert tuple(tensor.shape) == (3, 240, 480)
