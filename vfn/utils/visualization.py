import numpy as np
import torch

from typing import List, Union, Tuple
from PIL import Image, ImageColor, ImageDraw
from torchvision.transforms import ToPILImage


COLOR = {
    'sliding windows': ImageColor.getrgb('lightgrey'),
    'groundtruth': ImageColor.getrgb('lightpink'),
    'predict': ImageColor.getrgb('greenyellow')
}


def plot_bbox(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        bboxes: List[List[int]],
        bbox_type: str,
) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = ToPILImage()(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # here `image` is instance of `PIL.Image`

    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        draw.rectangle([x, y, x1, y1], outline=COLOR[bbox_type], width=2)

    return image


class Visualizer:

    pass
