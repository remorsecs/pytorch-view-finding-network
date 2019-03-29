from __future__ import absolute_import
from __future__ import print_function

import os
from tqdm import trange
from vfn.data.datasets.ioutils import download


class ImageDownloader(object):
    def __init__(self):
        super(ImageDownloader, self).__init__()

    @staticmethod
    def download(root_dir, image_urls):
        print('Downloading', len(image_urls), 'images to', root_dir)
        for i in trange(len(image_urls), ascii=True):
            url = image_urls[i]
            filepath = os.path.join(root_dir, os.path.basename(url))

            if not os.path.exists(filepath):
                download(url, filepath, verbose=False)
