import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from tqdm import tqdm, trange

import vfn.networks.backbones as backbones
from configs.parser import ConfigParser
from vfn.networks.models import ViewFindingNet
from vfn.networks.losses import ranknet_loss, svm_loss
from vfn.data.datasets.FlickrPro import FlickrPro
from vfn.data.datasets.FCDB import FCDB
from vfn.data.datasets.evaluation import ImageCropperEvaluator


def generate_crop_annos_by_sliding_window(image):
    crop_annos = []
    max_width, max_height = image.size

    for scale in range(5, 10):
        scale /= 10
        w, h = max_width * scale, max_height * scale
        dw, dh = max_width - w, max_height - h
        dw, dh = dw / 5, dh / 5

        for w_idx in range(5):
            for h_idx in range(5):
                x, y = w_idx * dw, h_idx * dh
                crop_annos.append([x, y, w, h])

    return crop_annos


def main():
    root_dir = ['FCDB', 'ICDB/All_Images']

    backbone = backbones.AlexNet()
    model = ViewFindingNet(backbone)
    model.load_state_dict(torch.load(kwargs['weight']))

    # FCDB
    dataset = FCDB(root_dir[0], download=False)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ground_truth, img_sizes, pred = [], [], []
    for i in trange(len(dataset)):
        filename, size, crop = dataset[i]
        ground_truth.append(crop)
        img_sizes.append(size)

        with Image.open(filename) as image:
            image = image.convert('RGB')
            crop_annos = generate_crop_annos_by_sliding_window(image)

            # add groundtruth
            crop_annos.append(crop)

            scores = []
            for crop_anno in crop_annos:
                x, y, w, h = crop_anno
                crop_image = image.crop((x, y, x+w, y+h))
                crop_image = data_transforms(crop_image)
                crop_image = crop_image.view((1, *crop_image.size()))

                score = model(crop_image)
                scores.append(score)

            scores = torch.tensor(scores)

        idx = scores.argmax().item()
        pred.append(crop_annos[idx])

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(ground_truth, pred, img_sizes)


if __name__ == '__main__':
    kwargs = dict(
        weight='ckpt/exp01.pth',
    )
    main()
