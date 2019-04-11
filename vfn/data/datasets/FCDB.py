from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import json
import shutil
from tqdm import trange
from torch.utils.data import Dataset
from vfn.data.datasets.ioutils import download
from vfn.data.datasets.image_downloader import ImageDownloader
from vfn.data.datasets.evaluation import ImageCropperEvaluator


class FCDB(Dataset):
    __meta_name = 'FCDB-%s.json'

    def __init__(self, root_dir, subset='testing', download=True):
        super(FCDB, self).__init__()
        assert subset in ['training', 'testing', 'all'], 'Unknown subset {}' % subset

        self.root_dir = root_dir
        self.subset = subset
        self.meta_file = os.path.join(root_dir, self.__meta_name % subset)

        if download:
            self._download(root_dir)

        self.img_list, self.img_sizes, self.annotations = self._fetch_metadata()
        self._check_integrity(root_dir)

    def get_all_items(self):
        return self.img_list, self.img_sizes, self.annotations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.img_sizes[index], self.annotations[index]

    def __str__(self):
        return 'FCDB dataset'

    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # download annotation file to root_dir
        url_fmt = \
            'https://raw.githubusercontent.com/yiling-chen/flickr-cropping-dataset/master/cropping_%s_set.json'
        if not os.path.exists(self.meta_file):
            print('Downloading FCDB annotation file...')
            if self.subset in ['training', 'testing']:
                anno_url = url_fmt % self.subset
                download(anno_url, os.path.join(self.root_dir, os.path.basename(anno_url)))
                shutil.move(os.path.join(self.root_dir, os.path.basename(anno_url)), self.meta_file)
                print()
            elif self.subset == 'all':
                # download both the training and testing sets
                anno_url = url_fmt % 'training'
                train_path = os.path.join(self.root_dir, os.path.basename(anno_url))
                download(anno_url, train_path)
                print()
                anno_url = url_fmt % 'testing'
                test_path = os.path.join(self.root_dir, os.path.basename(anno_url))
                download(anno_url, test_path)
                print()
                # Merge training and testing sets
                merge_dataset = json.load(open(train_path, 'r')) + json.load(open(test_path, 'r'))
                json.dump(merge_dataset, open(self.meta_file, 'w'))
                os.remove(train_path)
                os.remove(test_path)

        # Collect URLs and pass to ImageDownloader
        db = json.load(open(self.meta_file, 'r'))
        img_urls = [x['url'] for x in db]
        ImageDownloader.download(root_dir, img_urls)

    def _fetch_metadata(self):
        assert os.path.isfile(self.meta_file), "Metadata does not exist! Please download the FCDB dataset first!"

        print('Reading metadata...')
        db = json.load(open(self.meta_file, 'r'))
        img_list, img_sizes, annotations = [], [], []
        for i in trange(len(db)):
            # Some images might not be available on Flickr anymore, skip them
            img_path = os.path.join(self.root_dir, os.path.basename(db[i]['url']))
            if not os.path.exists(img_path):
                continue
            img_list.append(img_path)
            annotations.append(db[i]['crop'])
            height, width = cv2.imread(img_path).shape[:2]
            img_sizes.append((width, height))
        print('Unpacked', len(img_list), 'records.')

        return img_list, img_sizes, annotations

    def _check_integrity(self, root_dir):
        pass


def main():
    db = FCDB("../../../FCDB", subset='all')
    _, img_sizes, ground_truth = db.get_all_items()

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(ground_truth, ground_truth, img_sizes)


if __name__ == "__main__":
    main()
