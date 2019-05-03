from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
from tqdm import trange
from collections import defaultdict
from torch.utils.data import Dataset
from viewfinder_benchmark.data.ioutils import download, extract
from viewfinder_benchmark.data.evaluation import ImageCropperEvaluator


class ICDB(Dataset):
    __meta_name = 'cuhk_cropping.zip'

    def __init__(self, root_dir, subset=1, download=True):
        super(ICDB, self).__init__()
        # CUHK dataset contains three annotation subsets.
        assert subset in [1, 2, 3], 'Unknown subset: %d (Valid value: 1, 2, 3)' % subset

        self.subset = subset
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'All_Images')
        self.meta_file = os.path.join(root_dir, self.__meta_name)

        if download:
            self._download(root_dir)

        self.img_list, self.img_sizes, self.crops, self.category = self._fetch_metadata()
        self.img_groups, self.crop_groups, self. size_groups = self._group_metadata()
        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.img_sizes[index], self.crops[index], self.category[index]

    def __str__(self):
        return 'ICDB dataset'

    def get_metadata_by_group(self, label):
        assert label in self.img_groups.keys(), 'Unknown category %s' % label
        return self.img_groups[label], self.crop_groups[label], self.size_groups[label]

    def get_categories(self):
        return self.img_groups.keys()

    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # download annotation file to root_dir
        if not os.path.exists(self.meta_file):
            print('Downloading CUHK ICDB dataset...')
            anno_url = \
                'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/cuhk_cropping.zip'
            download(anno_url, self.meta_file)

        if not os.path.isdir(self.image_dir):
            if not os.path.isfile(os.path.join(self.root_dir, 'All_Images.zip')):
                print('\nExtracting dataset...')
                extract(self.meta_file, root_dir)

            print('Extracting images...')
            extract(os.path.join(self.root_dir, 'All_Images.zip'), self.root_dir)

    def _fetch_metadata(self):
        annotation_file = os.path.join(self.root_dir, 'Cropping parameters.txt')
        assert os.path.exists(annotation_file), 'Parameter file does not exist!'

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        print('Reading metadata...')
        num_images = round(len(lines) / 4)
        img_list, img_sizes, annotations, category = [], [], [], []
        for i in trange(num_images):
            label, filename = lines[i*4].strip().split('\\')
            crop = [int(x) for x in lines[i*4 + self.subset].split(' ')]
            # convert from (y1, y2, x1, x2) to (x, y, w, h) format
            annotations.append([crop[2], crop[0], crop[3] - crop[2], crop[1] - crop[0]])
            img_path = os.path.join(self.image_dir, filename)
            img_list.append(img_path)
            height, width = cv2.imread(os.path.join(self.image_dir, filename)).shape[:2]
            img_sizes.append((width, height))
            category.append(label)

        return img_list, img_sizes, annotations, category

    def _group_metadata(self):
        img_groups, crop_groups, size_groups = defaultdict(list), defaultdict(list), defaultdict(list)
        for img, size, crop, label in zip(self.img_list, self.img_sizes, self.crops, self.category):
            img_groups[label].append(img)
            crop_groups[label].append(crop)
            size_groups[label].append(size)
        return img_groups, crop_groups, size_groups

    def _check_integrity(self, root_dir):
        pass


def main():
    db = ICDB("../../../ICDB")
    print(db[0])

    _, crops, sizes = db.get_metadata_by_group('animal')
    print(db.get_categories())

    evaluator = ImageCropperEvaluator()
    # evaluate ground truth, this should get perfect results
    evaluator.evaluate(crops, crops, sizes)


if __name__ == "__main__":
    main()
