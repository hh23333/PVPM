from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

from torchreid.data.datasets import ImageDataset
from torchreid.utils import read_image
import cv2
import numpy as np

class Paritial_REID(ImageDataset):

    def __init__(self, root='', **kwargs):
        dataset_dir = 'Partial-REID_Dataset'
        self.root=osp.abspath(osp.expanduser(root))
        # self.dataset_dir = self.root
        data_dir = osp.join(self.root, dataset_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')
        self.query_dir=osp.join(self.data_dir, 'occluded_body_images')
        self.gallery_dir=osp.join(self.data_dir, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)
        super(Paritial_REID, self).__init__(train, query, gallery, **kwargs)
        self.load_pose = isinstance(self.transform, tuple)
        if self.load_pose:
            if self.mode == 'query':
                self.pose_dir = osp.join(self.data_dir, 'occluded_body_pose')
            elif self.mode=='gallery':
                self.pose_dir = osp.join(self.data_dir, 'whole_body_pose')
            else:
                self.pose_dir=''


    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path,'*.jpg'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)

        if self.load_pose:
            img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
            pose_pic_name = img_name + '_pose_heatmaps.png'
            pose_pic_path = os.path.join(self.pose_dir, pose_pic_name)
            pose = cv2.imread(pose_pic_path, cv2.IMREAD_GRAYSCALE)
            pose = pose.reshape((pose.shape[0], 56, -1)).transpose((0,2,1)).astype('float32')
            pose[:,:,18:] = np.abs(pose[:,:,18:]-128)
            img, pose = self.transform[1](img, pose)
            img = self.transform[0](img)
            return img, pid, camid, img_path, pose
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid, img_path
