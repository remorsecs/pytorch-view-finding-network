import argparse
import functools
import multiprocessing as mp
import pickle
import os
import urllib.request

from multiprocessing import freeze_support, Pool

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Download the dataset into a specific folder.')
    parser.add_argument('-d', '--data_root', default='datasets/data/FlickrImageCrop/', type=str,
                        help='the path to the dataset')
    parser.add_argument('-p', '--pkl_path', default='datasets/data/db_21K.pkl', type=str,
                        help='the path to the `db_21K.pkl` file')
    parser.add_argument('-w', '--num_workers', default=mp.cpu_count(), type=int,
                        help='the number of workers that download dataset with multiprocessing')
    parser.add_argument('-i', '--ignore_exists', action='store_true',
                        help='ignore `data_root` exists error')
    return parser.parse_args()


def read_db(pkl_path):
    with open(pkl_path, 'rb') as file:
        return pickle.load(file)


def download_an_image(target_dir, url):
    filename = url.split('/')[-1]
    full_path = os.path.join(target_dir, filename)
    with open(full_path, 'wb') as f:
        f.write(read_image(url))


def read_image(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.data_root, exist_ok=args.ignore_exists)
    db = read_db(args.pkl_path)
    urls = [db[i]['url'] for i in range(0, len(db), 14)]

    freeze_support()
    with Pool(args.num_workers) as pool:
        with tqdm(total=len(db) // 14) as pbar:
            download_an_image_by_url = functools.partial(download_an_image, args.data_root)
            for _ in tqdm(pool.imap_unordered(download_an_image_by_url, urls),
                          desc=f'Downloading with {args.num_workers} worker(s)...', ascii=True):
                pbar.update()

# TODO: issue: the downloaded file is larger than original's
