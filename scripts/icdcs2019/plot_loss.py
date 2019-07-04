# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

#OUTPUTPATH='/media/sf_Shared_Data/tmp/p2psgd'
OUTPUTPATH='/media/sf_Shared_Data/tmp/icdcs2019'
LOGPATH='/home/shshi/gpuhome/repositories/p2p-dl/logs'
#MGDLOGPATH='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/alllogs/192.168.122.186/logs/'
MGDLOGPATH='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/mgd140/logs'

EPOCH = True
FONTSIZE=16

fig, ax = plt.subplots(1,1,figsize=(5,3.8))
#fig, ax = plt.subplots(1,1,figsize=(5,4.2))
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstman4': 'LSTM-an4'
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
sbd = seconds_between_datetimestring

def get_loss(line, isacc=False):
    if EPOCH:
        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('avg loss: ') > 0
        #if line.find('Epoch') > 0 and line.find('loss:') > 0 and not line.find('acc:')> 0:
        if line.find('Epoch') > 0 and valid: 
            items = line.split(' ')
            loss = float(items[-1])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    else:
        if line.find('average forward') > 0:
            items = line.split('loss:')[1]
            loss = float(items[1].split(',')[0])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    return None, None

def read_losses_from_log(logfile, isacc=False):
    f = open(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None 
    max_epochs = 140
    counter = 0
    for line in f.readlines():
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('Epoch') > 0 and valid:
        #if not time0 and line.find('INFO [  100]') > 0:
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if not time0:
                time0 = t
        if line.find('lr: ') > 0:
            try:
                lr = float(line.split(',')[-2].split('lr: ')[-1])
                lrs.append(lr)
            except:
                pass
        if line.find('average delay: ') > 0:
            delay = int(line.split(':')[-1])
            average_delays.append(delay)
        loss, t = get_loss(line, isacc)
        if loss and t:
            counter += 1
            losses.append(loss)
            times.append(t)
        if counter > max_epochs:
            break

        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        #    items = line.split(' ')
        #    loss = float(items[-1])
        #    #items = line.split('loss:')[1]
        #    #loss = float(items[1].split(',')[0])

        #    losses.append(loss)
        #    t = line.split(' I')[0].split(',')[0]
        #    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        #    times.append(t)
    f.close()
    if not EPOCH:
        average_interval = 10
        times = [times[t*average_interval] for t in range(len(times)/average_interval)]
        losses = [np.mean(losses[t*average_interval:(t+1)*average_interval]) for t in range(len(losses)/average_interval)]
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
    #losses = losses[0:45]
    #losses = losses[0:14]
    return losses, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    print('means: ', means)
    print('stds: ', stds)
    return means, stds

def plot_loss(logfile, label, isacc=False, title='ResNet-20'):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    if logfile.find('resnet50') > 0 or logfile.find('alexnet') > 0:
        losses = losses[0:45]
    print('losses: ', losses)
    norm_means, norm_stds = read_norm_from_log(logfile)

    #print('times: ', times)
    #print('Learning rates: ', lrs)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    #plt.plot(losses, label=label, marker='o')
    #plt.xlabel('Epoch')
    #plt.title('ResNet-20 loss')
    if isacc:
        ax.set_ylabel('Top-1 Validation Accuracy')
    else:
        ax.set_ylabel('Training loss')
    #plt.title('ResNet-50')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    color = coloriter.next()
    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    ax.plot(range(0,len(losses)), losses, label=label, marker=marker, markerfacecolor='none', color=color)
    #ax.plot(range(1,len(losses)+1), losses, label=label, marker=marker, markerfacecolor='none', color=color)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    ax.set_xlabel('Epoch')
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    ax.grid(linestyle=':')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
        #for i in lr_indexes:
        #    if i < len(losses):
        #        ls = losses[i]
        #        ax.text(i, ls, 'lr=%f'%lrs[i])
    u.update_fontsize(ax, FONTSIZE)

def plot_loss_with_host(hostn, nworkers, hostprefix, baseline=False):
    if not baseline or nworkers == 64:
        port = 5922
    else:
        port = 5945
    for i in range(hostn, hostn+1):
        for j in range(2, 3):
            host='%s%d-%d'%(hostprefix, i, port+j)
            if baseline:
                logfile = './ad-sgd-%dn-%dw-logs/'%(nworkers/4, nworkers)+host+'.log'
            else:
                logfile = './%dnodeslogs/'%nworkers+host+'.log'
                if nworkers == 256 and hostn < 48:
                    host='%s%d.comp.hkbu.edu.hk-%d'%(hostprefix, i, port+j)
                    logfile = './%dnodeslogs/'%nworkers+host+'.log'
                #csr42.comp.hkbu.edu.hk-5922.log
                #logfile = './%dnodeslogs-w/'%nworkers+host+'.log'
            label = host+' ('+str(nworkers)+' workers)'
            if baseline:
                label += ' Baseline'
            plot_loss(logfile, label) 

def plot_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False, logpath=None):
    postfix='5922'
    if logpath is None:
        logpath = LOGPATH
        if hostname.find('MGD') >= 0:
            logpath = MGDLOGPATH
    if prefix.find('allreduce')>=0:
        postfix='0'
        if logpath.find('.186') > 0:
            postfix='124'
    if sparsity:
        logfile = '%s/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (logpath, prefix, dnn, nworkers, bs, lr, sparsity)
    elif nsupdate:
        logfile = '%s/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (logpath, prefix, dnn, nworkers, bs, lr, nsupdate)
    else:
        logfile = '%s/%s/%s-n%d-bs%d-lr%.4f' % (logpath, prefix, dnn, nworkers, bs, lr)
    if sg is not None:
        logfile += '-sg%.2f' % sg
    if density is not None:
        logfile += '-ds%s' % str(density)
    logfile += '/%s-%s.log' % (hostname, postfix)
    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    plot_loss(logfile, l, isacc=isacc, title=dnn) 

def resnet20():
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    #plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', '(Ref 1/4 data)', prefix='compression-modele',sparsity=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 4, 32, 0.5, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', sparsity=0.96)
    #plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', sparsity=0.98)
    #plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', sparsity=0.995)
    #plot_with_params('resnet20', 4, 32, 0.01, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 4, 32, 0.001, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+wait', prefix='baseline-wait-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.1, 'MGD', 'ADPSGD', prefix='baseline-modelmgd')
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modelhpcl', sparsity=0.9)
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modelhpcl', sparsity=0.95)
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modelhpcl', sparsity=0.99)
    #plot_with_params('resnet20', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.2, 'hpclgpu', 'ADPSGD+warmup', prefix='baseline-gwarmup-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.5, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.01, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 8, 32, 0.001, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.1, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.1, 'hpclgpu', 'ADPSGD+wait', prefix='baseline-wait-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.5, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.01, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.001, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 16, 32, 0.0005, 'hpclgpu', 'ADPSGD', prefix='baseline-modelhpcl')
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    #plot_with_params('resnet20', 8, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    #plot_with_params('resnet20', 16, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'AllReduce Comp s=2*std (~35x compression)', prefix='allreduce-comp-s0.9700-baseline-dc1-modeldebug')
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce Comp s=2*std', prefix='allreduce-comp-baseline-dc1-model-icdcs', nsupdate=1)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'AllReduce Comp s=3*std', prefix='allreduce-comp-baseline-gwarmup-dc1-model-icdcs', nsupdate=1)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu14', 'AllReduce Comp s=2*std (~35x compression)+warmup', prefix='allreduce-comp-baseline-gwarmup-dc1-model-icdcs', nsupdate=1)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu14', 'AllReduce Comp s=global top1% +warmup', prefix='allreduce-topk-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=2.0)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce local top1% ', prefix='allreduce-comp-baseline-gwarmup-dc1-model-icdcs-localtopk', nsupdate=1, sg=1.5)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce global top'+str(0.05*100)+'% ', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=1.5, density=0.05)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce global top'+str(0.01*100)+'% ', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=1.5, density=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce Comp s=global top'+str(0.1*100)+'% +warmup', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=1.5, density=0.1)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce global top'+str(0.04*100)+'% reduce', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=1.5, density=0.04)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu17', 'AllReduce global top[80->0.5]% reduce', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-dk', nsupdate=1, sg=1.5, density=0.04)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu17', 'AllReduce global top[80->1]% reduce', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-dk1', nsupdate=1, sg=1.5, density=0.04)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu20', 'AllReduce global top5% pb', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n', nsupdate=1, sg=1.5, density=0.05)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'AllReduce global top1% pb', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n', nsupdate=1, sg=1.5, density=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu17', 'AllReduce global top0.1% pb', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n', nsupdate=1, sg=1.5, density=0.001)
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'AllReduce global top0.1% pb+warmup', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu15', 'AllReduce global top0.01% pb+warmup', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.0001)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce global top0.05% pb+warmup', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.0005)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu10', 'AllReduce Comp s=sum top1% +warmup', prefix='allreduce-comp-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=2.0)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'AllReduce Comp s=2.8*std (~35x compression)+warmup', prefix='allreduce-comp-baseline-gwarmup-dc1-model-icdcs', nsupdate=1, sg=2.8)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')

def vgg16():
    #plot_with_params('vgg16', 4, 32, 0.1, 'gpu17', 'Allreduce', prefix='allreduce')
    #plot_with_params('vgg16', 4, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.95)
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.98)
    plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.01, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-wait-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'MGD', 'ADPSGD ', prefix='baseline-modelmgd', title='VGG16')
    plot_with_params('vgg16', 16, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.1, 'gpu20', 'ADPSGD ', prefix='baseline-modelk80', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.0005, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 8, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 16, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')

def mnistnet():
    plot_with_params('mnistnet', 1, 512, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 512, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 64, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')

def plot_one_worker():
    def _plot_with_params(bs, lr, isacc=True):
        logfile = './logs/resnet20/accresnet20-bs%d-lr%s.log' % (bs, str(lr))
        t = '(lr=%.4f, bs=%d)'%(lr, bs)
        plot_loss(logfile, t, isacc=isacc, title='resnet20') 
    _plot_with_params(32, 0.1)
    _plot_with_params(32, 0.01)
    _plot_with_params(32, 0.001)
    _plot_with_params(64, 0.1)
    _plot_with_params(64, 0.01)
    _plot_with_params(64, 0.001)
    _plot_with_params(128, 0.1)
    _plot_with_params(128, 0.01)
    _plot_with_params(128, 0.001)
    _plot_with_params(256, 0.1)
    _plot_with_params(256, 0.01)
    _plot_with_params(256, 0.001)
    _plot_with_params(512, 0.1)
    _plot_with_params(512, 0.01)
    _plot_with_params(512, 0.001)
    _plot_with_params(1024, 0.1)
    _plot_with_params(1024, 0.01)
    _plot_with_params(1024, 0.001)
    _plot_with_params(2048, 0.1)

def resnet50():
    plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'allreduce 4 GPUs', isacc=False, title='ResNet-50') 

    plot_with_params('resnet50', 8, 64, 0.01, 'gpu10', 'allreduce 8 GPUs', prefix='allreduce-debug')
    plot_with_params('resnet50', 8, 64, 0.01, 'gpu16', 'ADPSGD', prefix='baseline-dc1-modelk80')

def icdcs2019_convergence(network):
    # Convergence
    gtopk_name = r'gTop-$k$ S-SGD'
    dense_name = 'S-SGD'

    # resnet20
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu21', dense_name, prefix='allreduce', force_legend=True)
        plot_with_params(network, 4, 32, 0.1, 'gpu13', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'vgg16':
        # vgg16
        plot_with_params(network, 4, 128, 0.1, 'gpu13', dense_name, prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu10', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', dense_name, prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 256, 0.01, 'gpu18', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
        #plot_with_params(network, 8, 256, 0.01, 'gpu18', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'resnet50':
        plot_with_params(network, 8, 64, 0.01, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)
        plot_with_params(network, 4, 32, 0.01, 'gpu12', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
        #plot_with_params(network, 128, 256, 0.01, 'MGD', dense_name, prefix='allreduce-baseline-gwarmup-dc1-modelmgd', nsupdate=16, force_legend=True, logpath='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/alllogs/192.168.122.186/logs')
        #plot_with_params(network, 32, 256, 0.01, 'MGD', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-modelmgd-production', nsupdate=16, sg=2.5, density=0.001, force_legend=True)
        #plot_with_params(network, 32, 256, 0.01, 'MGD', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-modelmgd-debug', nsupdate=16, sg=2.5, density=0.001, force_legend=True, logpath='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/alllogs/192.168.122.101/logs')

def icdcs_camera_ready():
    # Convergence
    dense_name = 'Dense S-SGD'
    topk_name = r'Select $k$ from $k\times P$'
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', dense_name, prefix='allreduce', force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', topk_name, prefix='allreduce-comp-topk2-baseline-gwarmup-dc1-model-icdcs-cr', nsupdate=1, sg=2.5, density=0.001, force_legend=True)

def sensivities(network):
    if network == 'resnet20':
        density=0.001
        plot_with_params(network, 4, 32, 0.1, 'gpu13', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
        density=0.0005
        plot_with_params(network, 4, 32, 0.1, 'gpu10', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 32, 0.1, 'gpu15', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.0001, force_legend=True)
    elif network == 'vgg16':
        density=0.001
        plot_with_params(network, 4, 128, 0.1, 'gpu10', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0005
        plot_with_params(network, 4, 128, 0.1, 'gpu20', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 128, 0.1, 'gpu20', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)


def communication_speed():
    pass

def icdcs2019all():
    def convergence():
        #network = 'resnet20'
        #network = 'vgg16'
        network = 'alexnet'
        network = 'resnet50'
        icdcs2019_convergence(network=network)
        #icdcs_camera_ready()

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE)
        plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
        #plt.savefig('%s/%s_convergence.pdf' % (OUTPUTPATH, network))
        plt.savefig('%s/%s_convergence_cr.pdf' % (OUTPUTPATH, network))
        plt.show()


    def sensitivity():
        #network = 'resnet20'
        network = 'vgg16'
        #network = 'alexnet'
        #network = 'resnet50'
        sensivities(network=network)

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE)
        plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
        plt.savefig('%s/%s_sensitivity.pdf' % (OUTPUTPATH, network))
        plt.show()


    def communication_speed():
        pass

    convergence()
    #sensitivity()

if __name__ == '__main__':
    icdcs2019all()
