from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import json

from PIL import Image
from tqdm import trange
from torch.utils.data import Dataset
from vfn.data.datasets.evaluation import ImageCropperEvaluator
from vfn.data.datasets.image_downloader import download_images_from_urls
from vfn.data.datasets.ioutils import download


class FCDB(Dataset):
    __meta_name = 'FCDB-test.json'

    def __init__(self, root_dir, subset='test', download=True):
        super(FCDB, self).__init__()
        assert subset in ['train', 'test', 'all'], 'Unknown subset {}' % subset
        # TODO: handle 'train' and 'all' options

        self.root_dir = root_dir
        self.meta_file = os.path.join(root_dir, self.__meta_name)

        if download:
            self._download_metadata()

        self.filenames, self.annotations, self.urls = [], [], []
        self._fetch_metadata()

        if download:
            self._download_images()

        self.image_sizes = []
        self._fetch_image_sizes()
        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return self.filenames[index], self.image_sizes[index], self.annotations[index]

    def _download_metadata(self):
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)

        # download annotation file to root_dir
        if not os.path.exists(self.meta_file):
            print('Downloading FCDB annotation file...')
            anno_url =\
                'https://raw.githubusercontent.com/yiling-chen/flickr-cropping-dataset/master/cropping_testing_set.json'
            download(anno_url, self.meta_file)
            print('Done')

    def _download_images(self):
        print('Downloading FCDB images...')
        download_images_from_urls(self.root_dir, self.urls)
        print('Done.')

    def _fetch_metadata(self):
        assert os.path.isfile(self.meta_file), "Metadata does not exist! Please download the FCDB dataset first!"

        print('Reading metadata...')
        with open(self.meta_file, 'r') as f:
            db = json.load(f)

        for i in trange(len(db)):
            url = db[i]['url']
            self.urls.append(url)
            filename = os.path.basename(url)
            filename = os.path.join(self.root_dir, filename)
            self.filenames.append(filename)

            self.annotations.append(db[i]['crop'])
            # height, width = cv2.imread(os.path.join(self.root_dir, img_list[-1])).shape[:2]
            # img_sizes.append((width, height))
        print('Unpacked', len(db), 'records.')

    def _fetch_image_sizes(self):
        for i in trange(len(self.filenames)):
            image = Image.open(self.filenames[i]).convert('RGB')
            self.image_sizes.append(image.size)

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    db = FCDB("../../../FCDB")
    ground_truth, img_sizes = [], []
    for i in range(len(db)):
        filename, size, crop = db[i]
        ground_truth.append(crop)
        img_sizes.append(size)

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(ground_truth, ground_truth, img_sizes)
