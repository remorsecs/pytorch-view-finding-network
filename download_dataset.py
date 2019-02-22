import argparse
import multiprocessing as mp
import os
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description='Download the dataset into a specific folder.')
    parser.add_argument('-d', '--data_root', default='datasets/Flickr_21K/', type=str,
                        help='the path to the dataset')
    parser.add_argument('-p', '--pkl_root', default='datasets/db_21K.pkl', type=str,
                        help='the path to the `db_21K.pkl` file')
    parser.add_argument('-w', '--worker', default=mp.cpu_count(), type=int,
                        help='the number of workers that download dataset with multiprocessing')
    return parser.parse_args()


def download_dataset():
    args = parse_args()
    # TODO


def download_an_image(target_dir, url):
    filename = url.split('/')[-1]
    full_path = os.path.join(target_dir, filename)
    with open(full_path, 'wb') as f:
        f.write(read_image(url))


def read_image(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


if __name__ == '__main__':
    download_dataset()

