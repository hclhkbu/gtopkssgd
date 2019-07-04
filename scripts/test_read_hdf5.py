# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse, os
import glob
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

OUTPUTDIR='/tmp/imagenet_hdf5'

def test_read():
    h5file = os.path.join(OUTPUTDIR, 'imagenet-shuffled.hdf5')
    with h5py.File(h5file, 'r') as hf:
        imgs = hf['train_img'][0:10, ...]
        labels = hf["train_labels"][0:10]
        img = imgs[3]
        #img = img.transpose((1, 2, 0)) #np.moveaxis(imgs[0], 2, 0)
        #img = np.moveaxis(img, 2, 0)
        #img = img[...,[2,0,1]]
        print('labels: ', labels)
        print('image shape: ', img.shape)
        #cv2.imshow('h', img)
        plt.imshow(img)
    #cv2.waitKey(0)
    plt.show()

if __name__ == '__main__':
    test_read()
