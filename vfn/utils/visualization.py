import numpy as np
import torch

from enum import Enum
from typing import List, Union, Tuple
from PIL import Image, ImageColor, ImageDraw
from torchvision.transforms import ToPILImage


class ColorType(Enum):
    SLIDING_WINDOWS = ImageColor.getrgb('lightgrey')
    GROUNDTRUTH = ImageColor.getrgb('lightpink')
    PREDICT = ImageColor.getrgb('greenyellow')


def plot_bbox(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        bboxes: List[List[int]],
        bbox_type: ColorType,
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
        draw.rectangle([x, y, x1, y1], outline=bbox_type.value, width=2)

    return image


class Visualizer:

    pass
