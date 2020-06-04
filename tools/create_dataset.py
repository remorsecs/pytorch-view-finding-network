from __future__ import print_function

import sys
import cv2
import argparse
import shutil
import multiprocessing
import yaml
from gulpio.fileio import GulpIngestor
from gulpio.dataset import GulpDirectory
from gulpio.loader import DataLoader
from vfn.data.dataset import ImagePairListAdapter, ImagePairVisDataset
from vfn.data.FlickrPro import FlickrPro


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create image pair data for training trackers using GulpIO.')
    parser.add_argument('-w', '--workers', type=int, default=-1,
                        help="num of workers. -x uses (all - x) cores [-1 default].")
    parser.add_argument('-c', '--config', type=str, default='../configs/dataset.yml',
                        help='configuration file')
    parser.add_argument('-N', '--name', type=str, default='FlickrPro',
                        help='Dataset name [default: "FlickrPro"]')
    parser.add_argument('-r', '--root_folder', type=str,
                        help='root folder of GulpIO data')
    parser.add_argument('-n', '--images_per_chunk', type=int, default=2048,
                        help='number of images in one chunk [default: 2048]')
    parser.add_argument('-S', '--image_size', type=int, default=-1,
                        help='size of smaller edge of resized frames [default: -1 (no resizing)]')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Shuffle the dataset before ingestion [default: False]')
    parser.add_argument('-v', '--viz', action='store_true',
                        help='Visualize the dataset [default: False]')
    args = parser.parse_args()

    return args


def parse_config(config_path, name):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    if name not in config:
        raise Exception('Unrecognized dataset {}'.format(name))

    return config


def check_existing_dataset(data_path):
    gd = GulpDirectory(data_path)
    if gd.num_chunks > 0:
        print("Found existing dataset containing {} chunks in {}.".format(gd.num_chunks, data_path))
        print("Erase the existing dataset and create a new one? (y/n)")
        action = input()
        while action not in ['y', 'n']:
            action = input("Enter (y/n):")
        if action == 'y':
            shutil.rmtree(data_path)
        elif action == 'n':
            "Exiting. Nothing changed."
            sys.exit()


if __name__ == "__main__":
    args = parse_args()
    num_workers = args.workers

    if num_workers < 0:
        num_workers = multiprocessing.cpu_count() + num_workers

    config = parse_config(args.config, args.name)

    # image_list_file = args.image_list
    images_per_chunk = args.images_per_chunk
    img_size = args.image_size
    viz = args.viz

    if viz:
        dataset = ImagePairVisDataset(args.root_folder)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print("data size:", len(dataset))

        for img_full, img_crop in loader:
            cv2.imshow("Image Full", img_full[0])
            cv2.imshow("Image Crop", img_crop[0])
            key = cv2.waitKey()
            if key == 27:
                break
    else:
        for subset in ['train', 'val']:
            check_existing_dataset(config[args.name][subset]['gulpio_dir'])

            adapter = ImagePairListAdapter(
                FlickrPro(
                    config[args.name][subset]['root_dir'],
                    config[args.name][subset]['meta'],
                    config[args.name][subset]['download']
                ),
                # args.shuffle
            )

            ingestor = GulpIngestor(adapter, config[args.name][subset]['gulpio_dir'], images_per_chunk, num_workers)
            ingestor()  # call to trigger ingestion

