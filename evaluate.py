# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import logging
import torch
from dl_trainer import DLTrainer 
import argparse
from settings import logger, formatter

def model_average(trainers):
    trainer = trainers[0]
    own_state = trainer.net.state_dict()
    for name, param in own_state.items():
        for t in trainers[1:]:
            own_state[name] = own_state[name]+t.net.state_dict()[name]
    for name, param in own_state.items():
        own_state[name] = own_state[name]/len(trainers)
    trainer.net.load_state_dict(own_state)

def evaluate(model_path, dnn, dataset, data_dir, nepochs, allreduce=False):
    items = model_path.split('/')[-1].split('-')
    dnn = items[0]
    lr = float(items[-1][2:])
    batch_size = int(items[2][2:])
    #batch_size = 1 #int(items[2][2:])
    rank = 0 
    nworkers=1

    trainer = DLTrainer(rank, 1, dist=False, ngpus=1, batch_size=batch_size, is_weak_scaling=True, dataset=dataset, dnn=dnn, data_dir=data_dir, lr=lr, nworkers=nworkers)
    best_acc = 0.0
    start_epoch = 1
    for i in range(start_epoch, nepochs+1):
        filename = '%s-rank%d-epoch%d.pth' % (dnn, rank, i)
        fn = os.path.join(model_path, filename)
        if i == nepochs and not allreduce and False:
            trainers = []
            for j in range(nworkers):
                filename = '%s-rank%d-epoch%d.pth' % (dnn, j, i)
                fn = os.path.join(model_path, filename)
                tr = DLTrainer(rank, 1, dist=False, ngpus=1, batch_size=batch_size, is_weak_scaling=True, dataset=dataset, dnn=dnn, data_dir=data_dir, lr=lr, nworkers=nworkers)
                tr.load_model_from_file(fn)
                trainers.append(tr)
            model_average(trainers)
            trainer = trainers[0]
        else:
            trainer.load_model_from_file(fn)
        acc = trainer.test(i)
        if i == start_epoch:
            best_acc = acc
        else:
            if dnn in ['lstm', 'lstman4']: # the lower the better
                if best_acc > acc:
                    best_acc = acc
            else:
                if best_acc < acc:
                    best_acc = acc
        logger.info('Best validation accuracy or perprexity: %f', best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="p2pdl model evaluater")
    parser.add_argument('--model-path', type=str, help='Saved model path')
    parser.add_argument('--dnn', type=str, default='resnet20')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--nepochs', type=int, default=90, help='Number of epochs to evaluate')
    args = parser.parse_args()
    logfile = '%s/evaluate.log' % args.model_path
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    evaluate(args.model_path, args.dnn, args.dataset, args.data_dir, args.nepochs)


