from __future__ import absolute_import
import os.path as osp

from PIL import Image


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
        #00001368_00_0000.jpg
        fpath = fname
        
        if self.root is not None:
            #/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v1-reid/market1501/images/00001368_00_0000.jpg
            fpath = osp.join(self.root, fname)
          
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid
        """By returning two identical images with the same filename, person ID, and camera ID, the model can learn to recognize the same person
        regardless of the camera or time at which the images were taken."""  
    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname#00000420_00_0002.jpg
        #self.root /home/rdehghani/intra-inter-resnet/Person_ReIdentification/v1-reid/market1501/market1501/images
        #print("self.root",self.root)
        #exit(0)
        if self.root is not None:
            fpath = osp.join(self.root, fname)
            #/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v1-reid/market1501/market1501/images
            #fpath
            #/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v1-reid/market1501/market1501/images/00000420_00_0002.jpg
          
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
