import argparse
import torch
from torchvision.transforms import transforms

from viewfinder_benchmark.config.parser import ConfigParser
from viewfinder_benchmark.data.evaluation import ImageCropperEvaluator, generate_crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to config file (.yml)', default='../configs/example.yml')
    args = parser.parse_args()

    configs = ConfigParser(args.config)

    testsets = [
        configs.parse_FCDB(),
        configs.parse_ICDB(subset_selector=1),
        configs.parse_ICDB(subset_selector=2),
        configs.parse_ICDB(subset_selector=3),
    ]
    device = configs.parse_device()
    model = configs.parse_model().to(device)
    weight = torch.load(configs.configs['weight'], map_location=lambda storage, loc: storage)
    model.load_state_dict(weight)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for testset in testsets:
        evaluator = ImageCropperEvaluator(model, testset, device, data_transforms)
        print('Evaluate on {}'.format(testset))
        print('Average overlap ratio: {:.4f}'.format(evaluator.intersection_over_union))
        print('Average boundary displacement: {:.4f}'.format(evaluator.boundary_displacement))
        print('Alpha recall: {:.4f}'.format(evaluator.alpha_recall))
        print('Total image evaluated: {}'.format(evaluator.num_evaluated_images))


if __name__ == '__main__':
    main()
