from __future__ import absolute_import
from __future__ import print_function

import os
from torch.utils.data import Dataset


class ICDB(Dataset):
    def __init__(self, root_dir, download=True):
        super(ICDB, self).__init__()

        self.root_dir = root_dir
        if download:
            self._download(root_dir)

        self._check_integrity(root_dir)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

    def _fetch_metadata(self):
        pass

    def _check_integrity(self, root_dir):
        pass
