from __future__ import absolute_import
import os.path as osp

from PIL import Image

"""This code defines two classes, Preprocessor and BothPreprocessor,
that are used for pre-processing image datasets before they are fed into
a deep learning model for training or testing.

The Preprocessor class takes a dataset object, root directory path (optional), 
a transform object (optional), and a boolean variable mutual that determines whether to return 
a single image or two identical images with the same filename, person ID, and camera ID.
The __len__ method returns the length of the dataset, and the __getitem__ method retrieves
an image and its associated information at a given index.
If the mutual variable is True, the _get_mutual_item method is called to retrieve 
two identical images. Otherwise, the _get_single_item method is called to retrieve a single image.
Both methods load the image from the file path, apply the transform if it exists, and return 
the transformed image and its associated information.

The BothPreprocessor class is similar to the Preprocessor class, but it expects a dataset
object that contains additional information about the global person ID and camera-specific person ID. 
The _get_single_item method retrieves an image and its associated information 
at a given index, including the global person ID and camera-specific person ID. The returned information 
can be used for person re-identification tasks, where the goal is to match images of the same person across
different cameras."""
class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        # if isinstance(indices, (tuple, list)):
        #     return [self._get_single_item(index) for index in indices]
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)
        """The _get_single_item method takes an index, retrieves the corresponding image filename, person ID, and camera ID from the dataset, loads the image 
        from the file path, applies the transform if it exists, and returns the transformed image and its associated information."""
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
        """By returning two identical images with the same filename, person ID, and camera ID, the model can learn to recognize the same person
        regardless of the camera or time at which the images were taken."""  
    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, fname, pid, camid

class BothPreprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(BothPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        # if isinstance(indices, (tuple, list)):
        #     return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, global_pid, cam_pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, global_pid, cam_pid, camid
