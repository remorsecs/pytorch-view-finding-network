from __future__ import absolute_import


class ImageDownloader(object):
    def __init__(self, root_dir, image_urls):
        super(ImageDownloader, self).__init__()

        self.root_dir = root_dir
        self.image_urls = image_urls

    def _download(self):
        pass
