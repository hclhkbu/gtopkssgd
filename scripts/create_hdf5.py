# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse, os
import glob
import h5py
import numpy as np
import cv2
import csv
import time
from random import shuffle

DATADIR='/tmp/imagenet/ILSVRC2012_dataset'
OUTPUTDIR='/tmp/imagenet_hdf5'

def get_list(listfile):
    files = []
    with open(listfile) as f:
        for l in f.readlines():
            files.append(l)
    return files

def gen_class_maps(trainfiles, valfiles):
    def _gen_maps(fl):
        map = {}
        label = 0
        for f in fl:
            cls = f.split('/')[-2]
            if not cls in map:
                map[cls] = label
                label += 1
        print('num of classes: ', len(map.keys()))
        return map
    class_labels = _gen_maps(trainfiles)
    train_labels = []
    val_labels = []
    for f in trainfiles:
        cls = f.split('/')[-2]
        label = class_labels[cls]
        train_labels.append(label)
    for f in valfiles:
        cls = f.split('/')[-2]
        label = class_labels[cls]
        val_labels.append(label)
    return train_labels, val_labels, class_labels

def convert(outputpath, output, nworkers, datadir):
    trainfiles = gen_list_from_folder(datadir, folder='train')#get_list(trainlist)
    shuffle(trainfiles) # shuffle list
    valfiles =  gen_list_from_folder(datadir, folder='val') #get_list(vallist) 
    print('Read train number of lines: %d' % len(trainfiles))
    print('Read val number of lines: %d' % len(valfiles))
    train_labels, val_labels, class_labels = gen_class_maps(trainfiles, valfiles)
    with open(os.path.join(outputpath, 'imagenet_label_mapping.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for k in class_labels:
            l = str(class_labels[k])
            writer.writerow([k, l])

    h5file = os.path.join(outputpath, output)
    ntrains = len(trainfiles)
    nvals = len(valfiles)
    #train_shape = (ntrains, 256, 256, 3)
    #val_shape = (nvals, 256, 256, 3)
    train_shape = (ntrains, 224, 224, 3)
    val_shape = (nvals, 224, 224, 3)
    with h5py.File(h5file, 'w') as hf:
        hf.create_dataset("train_img", train_shape, np.uint8)
        hf.create_dataset("val_img", val_shape, np.uint8)
        hf.create_dataset("train_labels", (train_shape[0],), np.int16)
        hf["train_labels"][...] = train_labels
        hf.create_dataset("val_labels", (val_shape[0],), np.int16)
        hf["val_labels"][...] = val_labels

        s = time.time()
        for i in range(ntrains):
            if i % 1000 == 0 and i > 1:
                print('Train data: {}/{}, time used: {}'.format(i, ntrains, (time.time()-s)))
                s = time.time()

            f = trainfiles[i]
            img = cv2.imread(f)
            img = cv2.resize(img, (train_shape[1], train_shape[2]), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = np.rollaxis(img, 2)
            hf["train_img"][i, ...] = img[None]

        s = time.time()
        for i in range(nvals):
            if i % 1000 == 0 and i > 1:
                print('val data: {}/{}, time used: {}'.format(i, ntrains, (time.time()-s)))
                s = time.time()

            f = valfiles[i]
            img = cv2.imread(f)
            img = cv2.resize(img, (train_shape[1], train_shape[2]), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = np.rollaxis(img, 2)
            hf["val_img"][i, ...] = img[None]
           

def gen_list_from_folder(path, folder='train'):
    f = '%s/%s/*/*.JPEG'%(path, folder)
    print(f)
    addrs = glob.glob(f)
    return addrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert the ImageNet2012 dataset to the HDF5 format")
    parser.add_argument('--outputpath', type=str, default=OUTPUTDIR)
    parser.add_argument('--output', type=str, default='imagenet-shuffled-224.hdf5')
    parser.add_argument('--nworkers', type=int, default=1, help='Multiple threads supported')
    parser.add_argument('--datadir', type=str, default=DATADIR, help='Specify the dataset path, e.g., %s' % DATADIR)
    args = parser.parse_args()
    convert(args.outputpath, args.output, args.nworkers, args.datadir)
