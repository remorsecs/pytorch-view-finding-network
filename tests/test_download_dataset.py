import os


from unittest import TestCase

from PIL import Image, ImageChops

from vfn.data.datasets.download_dataset import download_an_image


class TestDownloadAnImage(TestCase):
    def test_file_exists(self):
        url = 'https://farm6.staticflickr.com/5326/17704422191_400d428d48_c.jpg'
        target_dir = './'
        filename = url.split('/')[-1]
        full_path = os.path.join(target_dir, filename)

        download_an_image(target_dir, url)
        try:
            self.assertTrue(os.path.isfile(full_path))
        finally:
            os.remove(full_path)

    def test_file_is_equal_to_origin(self):
        url = 'https://farm6.staticflickr.com/5326/17704422191_400d428d48_c.jpg'
        target_dir = './'
        filename = url.split('/')[-1]
        full_path = os.path.join(target_dir, filename)

        download_an_image(target_dir, url)
        images = [
            Image.open(filename).convert('RGB'),
            Image.open('test_' + filename).convert('RGB'),
        ]
        try:
            # Ref: http://effbot.org/zone/pil-comparing-images.htm
            self.assertIsNone(ImageChops.difference(*images).getbbox())
        finally:
            os.remove(full_path)
