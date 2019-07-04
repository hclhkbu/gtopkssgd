# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import h5py
import numpy as np

class DatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, hdf5fn, t, transform=None, target_transform=None):
        """
        t: 'train' or 'val'
        """
        super(DatasetHDF5, self).__init__()
        self.hf = h5py.File(hdf5fn, 'r', libver='latest', swmr=True)
        self.t = t
        self.n_images= self.hf['%s_img'%self.t].shape[0]
        self.dlabel = self.hf['%s_labels'%self.t][...]
        self.d = self.hf['%s_img'%self.t]
        self.transform = transform
        self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.d[index, ...]
        target = self.dlabel[index]
        return img, np.int64(target)

    def __getitem__(self, index):
        img, target = self._get_dataset_x_and_target(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.n_images
