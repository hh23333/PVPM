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


class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)
        self.load_pose = isinstance(self.transform, tuple)
        if self.load_pose:
            self.train_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_train')
            self.gallery_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_test')
            self.query_pose_dir = osp.join(self.data_dir, 'query_pose')
            if self.mode == 'train':
                self.pose_dir = self.train_pose_dir
            elif self.mode == 'query':
                self.pose_dir = self.query_pose_dir
            elif self.mode == 'gallery':
                self.pose_dir = self.gallery_pose_dir
            else:
                raise ValueError('Invalid mode. Got {}, but expected to be '
                                 'one of [train | query | gallery]'.format(self.mode))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
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
#
class Market1501_simu_occluded(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.occluded_dir = osp.join(self.data_dir, 'simulate_occluded_image_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train, pid2label = self.process_dir(self.train_dir, relabel=True)
        train = self.process_occluded_dir(train, pid2label, self.occluded_dir)
        query, _ = self.process_dir(self.query_dir, relabel=False)
        gallery, _ = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501_simu_occluded, self).__init__(train, query, gallery, **kwargs)
        self.load_pose = isinstance(self.transform, tuple)
        if self.load_pose:
            self.train_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_train')
            self.gallery_pose_dir = osp.join(self.data_dir, 'bounding_box_pose_test')
            self.query_pose_dir = osp.join(self.data_dir, 'query_pose')
            if self.mode == 'train':
                self.pose_dir = self.train_pose_dir
                self.occluded_pose_dir = osp.join(self.data_dir,'simulate_occluded_pose_train')
            elif self.mode == 'query':
                self.pose_dir = self.query_pose_dir
            elif self.mode == 'gallery':
                self.pose_dir = self.gallery_pose_dir
            else:
                raise ValueError('Invalid mode. Got {}, but expected to be '
                                 'one of [train | query | gallery]'.format(self.mode))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data, pid2label

    def process_occluded_dir(self, data, pid2label, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        for img_path in img_paths:
            pid,camid = map(int,pattern.search(img_path).groups())
            if pid==-1:
                continue
            camid -=1
            pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data

    def __getitem__(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)

        if self.load_pose:
            img_name = '.'.join(img_path.split('/')[-1].split('.')[:-1])
            pose_pic_name = img_name + '_pose_heatmaps.png'
            if ('simulate' in img_path):
                pose_pic_path = os.path.join(self.occluded_pose_dir, pose_pic_name)
            else:
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
