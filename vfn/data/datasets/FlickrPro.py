from __future__ import absolute_import
from __future__ import print_function

import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from tqdm import trange
from vfn.data.datasets.image_downloader import ImageDownloader
from vfn.data.datasets.ioutils import download


class FlickrPro(Dataset):
    __meta_name = 'flickr_pro.pkl'

    def __init__(self, root_dir, download=True, transforms=None, is_train=True):
        super(FlickrPro, self).__init__()

        self.root_dir = root_dir
        self.meta_file = os.path.join(root_dir, self.__meta_name)
        self.transforms = transforms

        if download:
            self._download_metadata()

        self.img_list, self.annotations, self.urls = self._fetch_metadata()

        if download:
            self._download_images()

        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = os.path.join(self.root_dir, self.img_list[index])
        image_raw = Image.open(img_file).convert('RGB')
        x, y, w, h = self.annotations[index]
        image_crop = image_raw.crop((x, y, x+w, y+h))

        # resize
        image_raw = image_raw.resize((227, 227))
        image_crop = image_crop.resize((227, 227))

        if self.transforms:
            image_raw = self.transforms(image_raw)
            image_crop = self.transforms(image_crop)

        return image_raw, image_crop

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

        img_list, annotations, urls = [], [], []
        for i in trange(len(db)):
            if (i % 14) == 0:
                urls.append(db[i]['url'])

            img_list.append(os.path.basename(db[i]['url']))
            annotations.append(db[i]['crop'])

        print('Unpacked', len(db), 'records.')

        return img_list, annotations, urls

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    flickr_pro = FlickrPro("../../../raw_images")
    print(flickr_pro[0])
