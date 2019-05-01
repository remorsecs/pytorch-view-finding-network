from __future__ import print_function

import sys
import cv2
import argparse
import shutil
import multiprocessing
from gulpio.fileio import GulpIngestor
from gulpio.dataset import GulpDirectory
from gulpio.loader import DataLoader
from viewfinder_benchmark.data.dataset import ImagePairListAdapter, ImagePairVisDataset
from viewfinder_benchmark.data.FlickrPro import FlickrPro


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create image pair data for training trackers using GulpIO.')
    parser.add_argument('-w', '--workers', type=int, default=-1,
                        help="num of workers. -x uses (all - x) cores [-1 default].")
    parser.add_argument('-l', '--image_list', type=str,
                        help='text file which contains the list of images')
    parser.add_argument('-o', '--output_folder', type=str, default='gulpio',
                        help='output folder for GulpIO files')
    parser.add_argument('-r', '--root_folder', type=str,
                        help='root image folder')
    parser.add_argument('-n', '--images_per_chunk', type=int, default=64,
                        help='number of images in one chunk [default: 1024]')
    parser.add_argument('-S', '--image_size', type=int, default=-1,
                        help='size of smaller edge of resized frames [default: -1 (no resizing)]')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Shuffle the dataset before ingestion [default: False]')
    parser.add_argument('-v', '--viz', action='store_true',
                        help='Visualize the dataset [default: False]')
    args = parser.parse_args()

    return args


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
            sys.exit()


if __name__ == "__main__":
    args = parse_args()
    num_workers = args.workers

    if num_workers < 0:
        num_workers = multiprocessing.cpu_count() + num_workers

    image_list_file = args.image_list
    output_folder = args.output_folder
    images_per_chunk = args.images_per_chunk
    img_size = args.image_size
    root_path = args.root_folder
    shuffle = args.shuffle
    viz = args.viz

    if viz:
        dataset = ImagePairVisDataset(output_folder)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print("data size:", len(dataset))

        for img_full, img_crop in loader:
            cv2.imshow("Image Full", img_full[0])
            cv2.imshow("Image Crop", img_crop[0])
            key = cv2.waitKey()
            if key == 27:
                break
    else:
        check_existing_dataset(output_folder)

        adapter = ImagePairListAdapter(FlickrPro("raw_images/flickr_pro"))

        ingestor = GulpIngestor(adapter, output_folder, images_per_chunk, num_workers)
        ingestor()  # call to trigger ingestion
