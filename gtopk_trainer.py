# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import sys
import argparse, os
import settings
import utils
import logging
import distributed_optimizer as dopt
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from compression import compressors

from settings import logger, formatter
import horovod.torch as hvd
from tensorboardX import SummaryWriter
writer = None
relative_path = None


def robust_ssgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, compression=False, compressor='topk', nwpernode=1, sigma_scale=2.5, pretrain=None, density=0.01, prefix=None):
    global relative_path

    torch.cuda.set_device(dopt.rank()%nwpernode)
    rank = dopt.rank()
    if rank != 0:
        pretrain = None

    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix+'-ds%s'%str(density), pretrain=pretrain, tb_writer=writer)

    init_epoch = trainer.get_train_epoch()
    init_iter = trainer.get_train_iter()

    trainer.set_train_epoch(comm.bcast(init_epoch))
    trainer.set_train_iter(comm.bcast(init_iter))

    def _error_handler(new_num_workers, new_rank):
        logger.info('Error info catched by trainer')
        trainer.update_nworker(new_num_workers, new_rank)

    compressor = compressor if compression else 'none'
    compressor = compressors[compressor]
    is_sparse = compression

    logger.info('Broadcast parameters....')
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')

    norm_clip = None
    optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), compression=compressor, is_sparse=is_sparse, err_handler=_error_handler, layerwise_times=None, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer)

    trainer.update_optimizer(optimizer)

    iters_per_epoch = trainer.get_num_of_training_samples() / (nworkers * batch_size * nsteps_update)

    times = []
    NUM_OF_DISLAY = 100
    display = NUM_OF_DISLAY if iters_per_epoch > NUM_OF_DISLAY else iters_per_epoch-1
    logger.info('Start training ....')
    for epoch in range(max_epochs):
        hidden = None
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn == 'lstm':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f. Speed: %f images/s, current density: %f', time_per_iter, batch_size * nsteps_update / time_per_iter, optimizer.get_current_density())
                times = []
        optimizer.add_train_epoch()
        if settings.PROFILING_NORM:
            # For comparison purpose ===>
            fn = os.path.join(relative_path, 'gtopknorm-rank%d-epoch%d.npy' % (rank, epoch))
            fn2 = os.path.join(relative_path, 'randknorm-rank%d-epoch%d.npy' % (rank, epoch))
            fn3 = os.path.join(relative_path, 'upbound-rank%d-epoch%d.npy' % (rank, epoch))
            fn5 = os.path.join(relative_path, 'densestd-rank%d-epoch%d.npy' % (rank, epoch))
            arr = [] 
            arr2 = [] 
            arr3 = [] 
            arr4 = [] 
            arr5 = [] 
            for gtopk_norm, randk_norm, upbound, xnorm, dense_std in optimizer._allreducer._profiling_norms:
                arr.append(gtopk_norm)
                arr2.append(randk_norm)
                arr3.append(upbound)
                arr4.append(xnorm)
                arr5.append(dense_std)
            arr = np.array(arr)
            arr2 = np.array(arr2)
            arr3 = np.array(arr3)
            arr4 = np.array(arr4)
            arr5 = np.array(arr5)
            logger.info('[rank:%d][%d] gtopk norm mean: %f, std: %f', rank, epoch, np.mean(arr), np.std(arr))
            logger.info('[rank:%d][%d] randk norm mean: %f, std: %f', rank, epoch, np.mean(arr2), np.std(arr2))
            logger.info('[rank:%d][%d] upbound norm mean: %f, std: %f', rank, epoch, np.mean(arr3), np.std(arr3))
            logger.info('[rank:%d][%d] x norm mean: %f, std: %f', rank, epoch, np.mean(arr4), np.std(arr4))
            logger.info('[rank:%d][%d] dense std mean: %f, std: %f', rank, epoch, np.mean(arr5), np.std(arr5))
            np.save(fn, arr)
            np.save(fn2, arr2)
            np.save(fn3, arr3)
            np.save(fn5, arr5)
            # For comparison purpose <=== End
        optimizer._allreducer._profiling_norms = []
    optimizer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--compression', dest='compression', action='store_true')
    parser.add_argument('--compressor', type=str, default='topk', choices=compressors.keys(), help='Specify the compressors if \'compression\' is open')
    parser.add_argument('--sigma-scale', type=float, default=2.5, help='Maximum sigma scaler for sparsification')
    parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=90, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.set_defaults(compression=False)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    if args.compression:
        prefix = 'comp-' + args.compressor + '-' + prefix
    logdir = 'allreduce-%s/%s-n%d-bs%d-lr%.4f-ns%d-sg%.2f-ds%s' % (prefix, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, args.sigma_scale, str(args.density))
    relative_path = './logs/%s'%logdir
    utils.create_path(relative_path)
    rank = 0
    rank = dopt.rank()
    hvd.init()
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    
    logger.info('Interpreter: %s', sys.version)
    robust_ssgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.compression, args.compressor, args.nwpernode, args.sigma_scale, args.pretrain, args.density, prefix)
