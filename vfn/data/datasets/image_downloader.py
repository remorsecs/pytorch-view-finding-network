from __future__ import absolute_import


import os
import urllib.request
from tqdm import trange


# deprecated?
class ImageDownloader(object):
    def __init__(self, root_dir, image_urls):
        super(ImageDownloader, self).__init__()

        self.root_dir = root_dir
        self.image_urls = image_urls

    @staticmethod
    def _read_image(url):
        with urllib.request.urlopen(url) as response:
            return response.read()

    def download(self):
        for i in trange(len(self.image_urls), ascii=True):
            url = self.image_urls[i]
            filename = os.path.basename(url)
            filepath = os.path.join(self.root_dir, filename)

            if not os.path.exists(filepath):
                with open(filepath, 'wb') as image_file:
                    image_file.write(self._read_image(url))


def download(root_dir, image_urls):
    def _read_image(_url):
        with urllib.request.urlopen(_url) as response:
            return response.read()

    for i in trange(len(image_urls), ascii=True):
        url = image_urls[i]
        filename = os.path.basename(url)
        filepath = os.path.join(root_dir, filename)

        if not os.path.exists(filepath):
            with open(filepath, 'wb') as image_file:
                image_file.write(_read_image(url))
