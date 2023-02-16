from __future__ import print_function
import os.path as osp
import pandas as pd
import numpy as np

from ..serialization import read_json

#0002_c1s1_000451_03.jpg
def _pluck(identities, indices, relabel=False, validate_names=None):
    ret = []
    
    #the full name of all images (1501 identities)
    #print(identities[2])#a list containing full name of all images related to id=2#['00000002_00_0000.jpg', '00000002_00_0001.jpg', '00000002_00_0002.jpg',...]]
    #print("len(identities)",len(identities))#1502
    #print("len(indices)",len(indices))#the number of ids in each folder, for train, it is 651
    #print("indices[0]",indices)#a list containing ids of all images in a foolder,[2, 7, 10, 11, 12, 22, 27, 28, 30, 32, 35, 37, 42, 43, 46, 47, 52, 53, 56, 57, 59, 64, 65, 68, 69, 70, 76, 77, 81, 82, 86, 88, 90, 93, 95, 97, 98, 99, 104, 105, 106,...]] 
    
    
    
    #print all images for one identity, save all images of roya ex;
    #indecies is a list containing ids of all images
    #id is the first four digits of image from the left
    
    for index, pid in enumerate(indices):
        
        pid_images = identities[pid]
       # print("pid_images",pid_images)#a list containing 6 lists based on camid(two images with camid=00,01,02,03,[no images with camid=4],05) 
        #
       
        count=0
        #print("len pid_images",len(pid_images))#is 6
        for camid, cam_images in enumerate(pid_images):
            """
camid 0
len cam_images 5
camid 1
len cam_images 4
camid 2
len cam_images 6
camid 3
len cam_images 5
camid 4
len cam_images 9
camid 5
len cam_images 4
            """
           
            for fname in cam_images:
                
                if validate_names is not None:
                    if fname not in validate_names:
                        continue
                name = osp.splitext(fname)[0]
                x, y, z = map(int, name.split('_'))
                #if we have 00001493_02_0000.jpg as a full name of one image after removing text
                #x is 1493 which is pid
                #y is 2 which is camid
                #z is 0 which is img_id
                count+=1
               
            
                assert pid == x and camid == y
                if relabel:
                    
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    
   
    


 
    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

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
        self.num_cameras = self.meta['num_cameras']
        identities = self.meta['identities']
        gallery_names = self.meta.get('gallery_names', None)
        if gallery_names is not None:
            gallery_names = set(gallery_names)
        query_names = self.meta.get('query_names', None)
        if query_names is not None:
            query_names = set(query_names)
        #print("*************start pluck function************************")
        self.train = _pluck(identities, train_pids, relabel=True)
       # print("after train set*****************************")
        self.val = _pluck(identities, val_pids, relabel=True)
        #print("after val set*****************************")
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
       # print("after trainval set*****************************")
        self.query = _pluck(identities, self.split['query'], validate_names=query_names)
        #print("after query set*****************************")
        self.gallery = _pluck(identities, self.split['gallery'], validate_names=gallery_names)
       # print("after gallery set*****************************")
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
