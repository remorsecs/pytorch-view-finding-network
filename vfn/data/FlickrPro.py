from __future__ import absolute_import
from __future__ import print_function

import os
import pickle

from torch.utils.data import Dataset
from tqdm import trange
from vfn.data.image_downloader import ImageDownloader


class FlickrPro(Dataset):

    def __init__(self, root_dir, meta_file, download=False):
        super(FlickrPro, self).__init__()
        self.root_dir = root_dir
        self.meta_file = meta_file
        self._fetch_metadata()

        if download:
            self._download_images()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        return self.filenames[i], self.annotations[i]

    def get_all_items(self):
        return self.filenames, self.annotations, self.urls

    def _download_images(self):
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)

        print('Downloading FlickrPro images...')
        ImageDownloader.download(self.root_dir, self.urls)
        print('Done')

    def _fetch_metadata(self):
        assert os.path.isfile(self.meta_file), "Metadata does not exist! Please download the FlickrPro dataset first!"

        print('Reading metadata...')
        with open(self.meta_file, 'rb') as f:
            db = pickle.load(f)

        self.filenames = []
        self.annotations = []
        self.urls = []

        for i in trange(len(db) // 14):
            url = db[i*14]['url']
            self.urls.append(url)

            filename = os.path.join(self.root_dir, os.path.basename(url))

            for j in range(14):
                self.filenames.append(filename)
                self.annotations.append(db[i*14 + j]['crop'])

        # print(len(self.filenames), len(self.urls), len(self.annotations))
        print('Unpacked', len(db), 'records.')

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    print(os.getcwd())
    flickr_pro = FlickrPro(root_dir="../../datasets/FlickrPro/src",
                           meta_file="../../data/flickr_pro_train.pkl",
                           download=False)
    print(flickr_pro[0])
