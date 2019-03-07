from __future__ import absolute_import
from __future__ import print_function

import os
import json
from torch.utils.data import Dataset
from tqdm import trange
from image_cropper_evaluator import ImageCropperEvaluator

import sys
if sys.version_info[0] == 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class FCDB(Dataset):
    __meta_name = 'FCDB-test.json'

    def __init__(self, root_dir, subset='test', download=True):
        super(FCDB, self).__init__()
        assert subset in ['train', 'test', 'all'], 'Unknown subset {}' % subset

        self.root_dir = root_dir
        self.meta_file = os.path.join(root_dir, self.__meta_name)
        print(self.meta_file)

        if download:
            self._download(root_dir)

        self.img_list, self.annotations = self._fetch_metadata()
        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.annotations[index]

    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # download annotation file to root_dir
        if not os.path.exists(self.meta_file):
            print('Downloading FCDB annotation file...')
            anno_url = \
                'https://raw.githubusercontent.com/yiling-chen/flickr-cropping-dataset/master/cropping_testing_set.json'
            urlretrieve(anno_url, self.meta_file)
            print('Done')

        # collect URLs and pass to ImageDownloader
        # filter out unavailable images and save to new meta file

    def _fetch_metadata(self):
        assert os.path.isfile(self.meta_file), "Metadata does not exist! Please download the FCDB dataset first!"

        print('Reading metadata...')
        db = json.load(open(self.meta_file, 'r'))
        img_list, annotations = [], []
        for i in trange(len(db)):
            img_list.append(os.path.basename(db[i]['url']))
            annotations.append(db[i]['crop'])
        print('Unpacked', len(db), 'records.')

        return img_list, annotations

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    db = FCDB("../../../FCDB")
    ground_truth = []
    for i in range(len(db)):
        filename, crop = db[i]
        ground_truth.append(crop)

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(ground_truth, ground_truth)
