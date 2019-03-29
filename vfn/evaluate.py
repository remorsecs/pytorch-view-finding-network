import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import trange

from vfn.networks import backbones
from configs.parser import ConfigParser
from vfn.networks.models import ViewFindingNet
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


def evaluate_on(dataset, model, device):
    print('Evaluate on {} dataset.'.format(dataset))

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ground_truth, img_sizes, pred = [], [], []
    for i in trange(len(dataset), ascii=True):
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
                crop_image = crop_image.to(device)

                score = model(crop_image)
                scores.append(score)

            scores = torch.tensor(scores)

        idx = scores.argmax().item()
        pred.append(crop_annos[idx])

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(ground_truth, pred, img_sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='Path to config file (.yml)', default='../configs/DEFAULT.yml')
    args = parser.parse_args()

    configs = ConfigParser(args.config_file)

    datasets = [
        configs.parse_ICDB(),
        configs.parse_FCDB(),
    ]
    device = configs.parse_device()
    backbone = backbones.AlexNet()
    model = ViewFindingNet(backbone)
    model.load_state_dict(torch.load(configs.configs['weight']))
    model.to(device)
    for dataset in datasets:
        evaluate_on(dataset, model, device)


if __name__ == '__main__':
    main()
