from __future__ import print_function, absolute_import
import os.path as osp

"""
In summary, this code is creating a list of image tuples for the specific camera view,
and also counting the number of unique people in the cluster.
"""
class Cluster(object):
    def __init__(self, root, cluster_result_cam, cam_id):
        self.root = root
        self.train_set = []
        classes = []
        # iterating through the items of the cluster_result_cam dictionary using the .items() method.
        #fanme is the key in dict
        #pid is its value of the key in dict
        for fname, pid in cluster_result_cam.items():
            #append a tuple (fname, pid, cam_id) to the train_set list
            self.train_set.append((fname, pid, cam_id))
            # append the pid (person ID) to the classes list. This is done
            # so that the class can keep track of the number of unique person IDs in the cluster
            classes.append(pid)
            #using the set function to remove any duplicates to have unique pid
            #This provides the total number of unique people in the cluster.
        self.classes_num = len(set(classes))
    
    """The @property decorator in the images_dir method indicates
    that this method should be treated as a property rather than a method. """
    @property
    def images_dir(self):
        return osp.join(self.root, 'images')
