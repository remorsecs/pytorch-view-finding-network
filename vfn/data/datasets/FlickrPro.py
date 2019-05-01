from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
import random

from PIL import Image
from torch.utils.data import Dataset
from tqdm import trange
from vfn.data.datasets.image_downloader import ImageDownloader
from vfn.data.datasets.ioutils import download


class FlickrPro(Dataset):
    __meta_name = 'flickr_pro.pkl'

    def __init__(self, root_dir, download=True, transforms=None,):
        super(FlickrPro, self).__init__()
        self.root_dir = root_dir
        self.meta_file = os.path.join(root_dir, self.__meta_name)
        self.transforms = transforms
        self._download_metadata()
        self._fetch_metadata()

        if download:
            self._download_images()

        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with Image.open(self.filenames[i]) as image:
            raw_image = image.convert('RGB')
            x, y, w, h = self.annotations[i]
            crop_image = raw_image.crop((x, y, x+w, y+h))

            if self.transforms:
                raw_image = self.transforms(raw_image)
                crop_image = self.transforms(crop_image)

        return raw_image, crop_image

    def _download_metadata(self):
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)

        # download dataset.pkl to root_dir
        if not os.path.exists(self.meta_file):
            print('Downloading FlickrPro...')
            pkl_url = 'https://raw.githubusercontent.com/yiling-chen/view-finding-network/master/dataset.pkl'
            download(pkl_url, self.meta_file)
            print('Done')

    def _download_images(self):
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

        for i in trange(len(db)):
            url = db[i]['url']
            self.urls.append(url)

            filename = os.path.join(self.root_dir, os.path.basename(url))
            self.filenames.append(filename)

            crop = db[i]['crop']
            self.annotations.append(crop)

        print('Unpacked', len(db), 'records.')

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    print(os.getcwd())
    flickr_pro = FlickrPro("../../../raw_images/flickr_pro", download=False)
    # print(flickr_pro[0])
