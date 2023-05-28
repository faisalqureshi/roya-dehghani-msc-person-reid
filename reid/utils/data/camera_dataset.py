from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json

"""
This is a Python script that defines a class CameraDataset for loading a dataset of images for person re-identification (re-ID) tasks.

The class takes a root directory as input, where the images and metadata are stored. Upon initialization, the class sets various attributes, such as root, split_id, camera_id, meta, split, train, val, trainval, query, gallery, num_train_ids, num_val_ids, and num_trainval_ids.

The load method of the class reads the metadata and splits the data into different subsets: training, validation, trainval (training and validation combined), query, and gallery. The train, val, trainval, query, and gallery attributes are populated with a list of tuples, where each tuple contains the filename, person ID, and camera ID of an image.

The _pluck function is a helper function used by the load method to extract the required information from the metadata.

Finally, the _check_integrity method checks if the necessary files and directories exist in the root directory.

Overall, this script provides a convenient way to load data for person re-ID tasks.

"""
def _pluck(identities, indices, relabel=False, validate_names=None, camera_id=None):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                if validate_names is not None:
                    if fname not in validate_names:
                        continue
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    if camid == camera_id and camera_id is not None:
                        ret.append((fname, index, camid))
                    elif camera_id is None:
                        ret.append((fname, index, camid))
                else:
                    if camid == camera_id and camera_id is not None:
                        ret.append((fname, pid, camid))
                    elif camera_id is None:
                        ret.append((fname, pid, camid))
    return ret


class CameraDataset(object):
    def __init__(self, root, split_id=0, camera_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.camera_id = camera_id
    
    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        gallery_names = self.meta.get('gallery_names', None)
        if gallery_names is not None:
            gallery_names = set(gallery_names)
        query_names = self.meta.get('query_names', None)
        if query_names is not None:
            query_names = set(query_names)
        self.train = _pluck(identities, train_pids, relabel=True, camera_id=self.camera_id)
        self.val = _pluck(identities, val_pids, relabel=True, camera_id=self.camera_id)
        self.trainval = _pluck(identities, trainval_pids, relabel=True, camera_id=self.camera_id)
        self.query = _pluck(identities, self.split['query'], validate_names=query_names)
        self.gallery = _pluck(identities, self.split['gallery'], validate_names=gallery_names)
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
