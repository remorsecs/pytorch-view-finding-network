from __future__ import absolute_import
from __future__ import print_function

import os
import pathlib
from torch.utils.data import Dataset


class GAICD(Dataset):
    def __int__(self, root_dir, subset='train'):
        super(GAICD, self).__init__()
        assert subset in ['train', 'test'], 'Unknown subset {}' % subset

        self.root_dir = root_dir
        self.subset = subset

        self.img_list, self.annotations = self._fetch_metadata()

    def get_all_items(self):
        pass

    def __len__(self):
        return len(self.img_list)

    def __str__(self):
        return 'GAICD dataset'

    def _fetch_metadata(self):
        annotation_path = os.path.join(self.root_dir, 'annotations', self.subset)
        image_path = os.path.join(self.root_dir, 'images', self.subset)
        assert os.path.exists(annotation_path), 'Cannot find {}' % annotation_path
        assert os.path.exists(image_path), 'Cannot find {}' % image_path

        filenames = list()
        annotations = list()

        for filename in pathlib.Path(annotation_path).glob('*.txt'):
            with open(filename, 'r') as f:
                annotation_text = f.readlines()

            for line in annotation_text:
                items = line.split(' ')
                bbox = [int(x) for x in items[:-1]]
                filenames.append(filename)
                annotations.append(bbox)

        return filenames, annotations
