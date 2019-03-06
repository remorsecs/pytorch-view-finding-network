from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
from torch.utils.data import Dataset
from tqdm import trange

import sys
if sys.version_info[0] == 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class FlickrPro(Dataset):
    __meta_name = 'flickr_pro.pkl'

    def __init__(self, root_dir, download=True):
        super(FlickrPro, self).__init__()

        self.root_dir = root_dir
        self.meta_file = os.path.join(root_dir, self.__meta_name)

        if download:
            self._download(root_dir)

        self.img_list, self.annotations = self._fetch_metadata()
        self._check_integrity(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        pass

    # TODO: write the download code
    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # download dataset.pkl to root_dir
        if not os.path.exists(self.meta_file):
            print('Downloading FlickrPro...')
            pkl_url = 'https://raw.githubusercontent.com/yiling-chen/view-finding-network/master/dataset.pkl'
            urlretrieve(pkl_url, self.meta_file)
            print('Done')

        # collect URLs and pass to ImageDownloader

    def _fetch_metadata(self):
        assert os.path.isfile(self.meta_file), "Metadata does not exist! Please download the FlickrPro dataset first!"

        print('Reading metadata...')
        db = pickle.load(open(self.meta_file, 'rb'))
        img_list, annotations = [], []
        for i in trange(len(db)):
            img_list.append(os.path.basename(db[i]['url']))
            annotations.append(db[i]['crop'])
        print('Unpacked', len(db), 'records.')

        return img_list, annotations

    def _check_integrity(self, root_dir):
        pass


if __name__ == "__main__":
    flickr_pro = FlickrPro("../../../raw_images")
