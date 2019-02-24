import os

from unittest import TestCase

from download_dataset import download_an_image


class TestDownloadAnImage(TestCase):
    def test_file_exists(self):
        url = 'https://farm6.staticflickr.com/5326/17704422191_400d428d48_c.jpg'
        target_dir = './'
        filename = url.split('/')[-1]
        full_path = os.path.join(target_dir, filename)

        download_an_image(target_dir, url)
        self.assertTrue(os.path.isfile(full_path))
        os.remove(full_path)
