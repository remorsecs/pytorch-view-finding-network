import cv2
import numpy as np
import torch
from torchvision.transforms import transforms as T
from tqdm import tqdm



class ImageCropperEvaluator(object):

    def __init__(self, model, dataset, device, transforms=None, generate_crops_fn=None):
        with torch.no_grad():
            self._predict(model, dataset, device, transforms, generate_crops_fn)

        self._intersection_over_union_value = None
        self._mean_intersection_over_union_value = None
        self._mean_boundary_displacement_value = None
        self._alpha_recall_value = None

    @property
    def num_evaluated_images(self):
        return len(self._ground_truths)

    @property
    def intersection_over_union(self):
        if self._mean_intersection_over_union_value is None:
            self._init_intersection_over_union_value()

        return self._mean_intersection_over_union_value

    def _init_intersection_over_union_value(self):
        x1, y1, w1, h1 = self._ground_truths.t()
        x2, y2, w2, h2 = self._predictions.t()
        inter_w = torch.min(x1 + w1, x2 + w2) - torch.max(x1, x2)
        inter_w.clamp_min_(0.0)
        inter_h = torch.min(y1 + h1, y2 + h2) - torch.max(y1, y2)
        inter_h.clamp_min_(0.0)
        intersection = inter_w * inter_h
        union = (w1 * h1) + (w2 * h2) - intersection
        self._intersection_over_union_value = intersection / union
        self._mean_intersection_over_union_value = torch.mean(self._intersection_over_union_value).cpu()

    @property
    def boundary_displacement(self):
        if self._mean_boundary_displacement_value is None:
            w, h = self._image_sizes.t()

            x11, y11, w1, h1 = self._ground_truths.t()
            y12 = y11 + h1
            x12 = x11 + w1

            x21, y21, w2, h2 = self._predictions.t()
            y22 = y21 + h2
            x22 = x21 + w2

            x_displacement = (torch.abs(x11 - x21) + torch.abs(x12 - x22)) / w
            y_displacement = (torch.abs(y11 - y21) + torch.abs(y12 - y22)) / h
            self._mean_boundary_displacement_value = torch.mean((x_displacement + y_displacement) / 4).cpu()

        return self._mean_boundary_displacement_value

    @property
    def alpha_recall(self, alpha=0.75):
        if self._alpha_recall_value is None:
            if self._mean_intersection_over_union_value is None:
                self._init_intersection_over_union_value()

            self._alpha_recall_value = torch.mean(100 * (self._intersection_over_union_value > alpha).float()).cpu()

        return self._alpha_recall_value

    def _predict(self, model, dataset, device, transforms, generate_crops_fn):
        model.eval()
        model.to(device)
        if transforms is None:
            transforms = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
        if generate_crops_fn is None:
            generate_crops_fn = generate_crops

        image_sizes = []
        predictions = []
        ground_truths = []
        for filename, image_size, ground_truth in tqdm(dataset):
            image_sizes.append(image_size)
            ground_truths.append(ground_truth)

            image = cv2.imread(filename)
            image = image[..., [2, 1, 0]]
            width, height = image_size
            crops = [ground_truth] + generate_crops_fn(width, height)
            crop_images = self._generate_crop_images(image, crops, transforms).to(device)
            scores = model(crop_images)
            idx = scores.argmax().item()

            predictions.append(crops[idx])

        self._image_sizes = torch.tensor(image_sizes, dtype=torch.float32, device=device)
        self._predictions = torch.tensor(predictions, dtype=torch.float32, device=device)
        self._ground_truths = torch.tensor(ground_truths, dtype=torch.float32, device=device)

    @staticmethod
    def _generate_crop_images(image, crops, transforms):
        crop_images = []
        for crop in crops:
            x, y, w, h = crop
            crop_image = np.copy(image[y:y + h, x:x + w, :])
            crop_image = transforms(crop_image)
            crop_images.append(crop_image)

        return torch.stack(crop_images)


def generate_crops(width, height):
    crops = []

    for scale in range(5, 10):
        scale /= 10
        w, h = width * scale, height * scale
        dw, dh = width - w, height - h
        dw, dh = dw / 5, dh / 5

        for w_idx in range(5):
            for h_idx in range(5):
                x, y = w_idx * dw, h_idx * dh
                crops.append([int(x), int(y), int(w), int(h)])

    return crops
