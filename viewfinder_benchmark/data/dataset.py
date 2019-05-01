import cv2
import numpy as np

from torch.utils.data import Dataset
from gulpio.adapters import AbstractDatasetAdapter
from gulpio.utils import ImageNotFound
from gulpio.dataset import GulpDirectory, GulpIOEmptyFolder


class ImagePairListAdapter(AbstractDatasetAdapter):
    def __init__(self, src_dataset):
        self.source_dataset = src_dataset
        self.data = self._parse_dataset()
        import random
        random.shuffle(self.data)

    def __len__(self):
        # return len(self.flickr_pro)
        return 500

    def _parse_dataset(self):
        filenames, annotations, _ = self.source_dataset.get_all_items()
        data = []
        for i, (filename, annotation) in enumerate(zip(filenames[:500], annotations[:500])):
            data.append({'id': i,
                         'filename': filename,
                         'annotation': annotation})
        return data

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        slice_data = self.data[slice_element]
        for item in slice_data:
            try:
                # print(item['filename'])
                image = cv2.imread(item['filename'])
                img_full = np.copy(image)
                x, y, w, h = item['annotation']
                img_crop = np.copy(image[y:y+h, x:x+w, :])
                # print(img_full.shape, img_crop.shape)
                if 0 in img_crop.shape:
                    print(img_full.shape)
                    print(item['annotation'])
                    continue
            except:
                print("Failed to read {}!".format(item['filename']))
                continue  # skip the item if image is not readable
            # Always encapsulate your data into a dict with (id, meta, frames) keys
            # which will be processed by gulpio ChunkWriter
            result = {'meta': item,
                      'frames': [img_full, img_crop],
                      'id': item['id']}
            yield result


class ImagePairDataset(Dataset):
    def __init__(self, data_path, transforms):
        """Simple image pair data loader for GulpIO format.
            Args:
                data_path (str): path to GulpIO dataset folder
                is_va (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                target_transform (func): performs preprocessing on labels if
            defined. Default is None.
        """
        self.gd = GulpDirectory(data_path)
        self.num_chunks = self.gd.num_chunks
        self.transforms = transforms
        if self.num_chunks == 0:
            raise (GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                     "of: ".format(data_path)))

        self.data_path = data_path
        print("Found {} chunks in {}".format(self.num_chunks, self.data_path))
        self.items = list(self.gd.merged_meta_dict.items())

    def __getitem__(self, index):
        """
        With the given index, it fetches frames. This function is called
        by PyTorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        item_id, item_info = self.items[index]
        img, meta = self.gd[item_id]
        img_full, img_crop = img

        if self.transforms:
            img_full = self.transforms(img_full)
            img_crop = self.transforms(img_crop)

        return img_full, img_crop

    def __len__(self):
        return len(self.items)


class ImagePairVisDataset(Dataset):
    def __init__(self, data_path):
        """Simple image pair data loader for GulpIO format.
            Args:
                data_path (str): path to GulpIO dataset folder
        """
        self.gd = GulpDirectory(data_path)
        self.num_chunks = self.gd.num_chunks
        if self.num_chunks == 0:
            raise (GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                     "of: ".format(data_path)))

        self.data_path = data_path
        print("Found {} chunks in {}".format(self.num_chunks, self.data_path))
        self.items = list(self.gd.merged_meta_dict.items())

    def __getitem__(self, index):
        """
        With the given index, it fetches frames. This function is called
        by PyTorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        item_id, item_info = self.items[index]
        img, meta = self.gd[item_id]
        img_full, img_crop = img

        img_full = cv2.cvtColor(img_full, cv2.COLOR_RGB2BGR)
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)

        return img_full, img_crop

    def __len__(self):
        return len(self.items)
