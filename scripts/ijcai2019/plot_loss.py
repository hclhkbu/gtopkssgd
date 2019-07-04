# -*- coding: utf-8 -*-
from __future__ import print_function
from matplotlib import rcParams
FONT_FAMILY='DejaVu Serif'
rcParams["font.family"] = FONT_FAMILY 
from mpl_toolkits.axes_grid.inset_locator import inset_axes
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
fixed_colors = {
        'S-SGD': '#ff3300',
        'ssgd': '#ff3300',
        'gTopK': '#009900',
        'blue': 'b',
        0.001: 'C2',
        0.002: 'C5',
        0.00025: 'C3',
        0.0001: 'C0',
        0.00005: 'C1',
        0.00001: 'C4',
        }

OUTPUTPATH='/tmp/ijcai2019'
LOGHOME='/tmp/logs'

FONTSIZE=14
HOSTNAME='localhost'
num_batches_per_epoch = None
global_max_epochs=150
global_density=0.001
#NFIGURES=4;NFPERROW=2
NFIGURES=6;NFPERROW=2
#NFIGURES=1;NFPERROW=1
#FIGSIZE=(5*NFPERROW,3.8*NFIGURES/NFPERROW)
PLOT_NORM=False
PLOT_NORM=True
if PLOT_NORM:
    #FIGSIZE=(5*NFPERROW,3.1*NFIGURES/NFPERROW)
    FIGSIZE=(5*NFPERROW,3.2*NFIGURES/NFPERROW)
else:
    #FIGSIZE=(5*NFPERROW,2.9*NFIGURES/NFPERROW)
    FIGSIZE=(5*NFPERROW,3.0*NFIGURES/NFPERROW)

fig, group_axs = plt.subplots(NFIGURES/NFPERROW, NFPERROW,figsize=FIGSIZE)
if NFIGURES > 1 and PLOT_NORM:
    ax = None
    group_axtwins = []
    for i in range(NFIGURES/NFPERROW):
        tmp = []
        for a in group_axs[i]:
            tmp.append(a.twinx())
        group_axtwins.append(tmp)
    global_index = 0
else:
    ax = group_axs
    ax1 = ax
    global_index = None
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstm': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
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
    valid = line.find('val acc: ') > 0 if isacc else line.find('loss: ') > 0
    if line.find('Epoch') > 0 and valid: 
        items = line.split(' ')
        loss = float(items[-1])
        t = line.split(' I')[0].split(',')[0]
        t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        return loss, t

def read_losses_from_log(logfile, isacc=False):
    global num_batches_per_epoch
    f = open(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None 
    max_epochs = global_max_epochs
    counter = 0
    for line in f.readlines():
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        if line.find('Epoch') > 0 and valid:
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
    f.close()
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
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

def plot_loss(logfile, label, isacc=False, title='ResNet-20', fixed_color=None):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)

    print('times: ', times)
    print('losses: ', losses)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    if isacc:
        ax.set_ylabel('top-1 Validation Accuracy')
    else:
        ax.set_ylabel('training loss')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    if fixed_color:
        color = fixed_color
    else:
        color = coloriter.next()

    iterations = np.arange(len(losses)) 
    line = ax.plot(iterations, losses, label=label, marker=marker, markerfacecolor='none', color=color, linewidth=1)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    ax.set_xlabel('# of epochs')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
    u.update_fontsize(ax, FONTSIZE)
    return line


def plot_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False):
    global global_density
    global_density = density
    postfix='5922'
    color = None
    if prefix.find('allreduce')>=0:
        postfix='0'
    elif prefix.find('single') >= 0:
        postfix = None
    if sparsity:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (prefix, dnn, nworkers, bs, lr, sparsity)
    elif nsupdate:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, dnn, nworkers, bs, lr, nsupdate)
    else:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f' % (prefix, dnn, nworkers, bs, lr)
    if sg is not None:
        logfile += '-sg%.2f' % sg
    if density is not None:
        logfile += '-ds%s' % str(density)
        color = fixed_colors[density]
    else:
        color = fixed_colors['S-SGD']
    if postfix is None:
        logfile += '/%s.log' % (hostname)
    else:
        logfile += '/%s-%s.log' % (hostname, postfix)
    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    line = plot_loss(logfile, l, isacc=isacc, title=dnn, fixed_color=color) 
    return line

def plot_group_norm_diff():
    global ax
    networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    networks = ['vgg16', 'resnet20', 'alexnet', 'resnet50', 'lstm', 'lstman4']
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        ax = group_axs[ax_row][ax_col]
        ax1 = group_axtwins[ax_row][ax_col]
        plts = plot_norm_diff(ax1, network)
    lines, labels = ax.get_legend_handles_labels()
    STNAME
    lines2, labels2 = ax1.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, ncol=4, loc='upper center', fontsize=FONTSIZE, frameon=True)
    plt.subplots_adjust(bottom=0.09, left=0.08, right=0.90, top=0.88, wspace=0.49, hspace=0.42)
    plt.savefig('%s/multiple_normdiff.pdf'%OUTPUTPATH)

def plot_norm_diff(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    density = 0.001
    nsupdate=1
    prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019'
    if network == 'lstm':
        network = 'lstm';bs =100;lr=30.0;epochs =40
    elif network == 'lstman4':
        network = 'lstman4';bs =8;lr=0.0002;epochs = 80
    elif network == 'resnet20':
        network = 'resnet20';bs =32;lr=0.1;epochs=140
    elif network == 'vgg16':
        network = 'vgg16';bs=128;lr=0.1;epochs=140
    elif network == 'alexnet':
        network = 'alexnet';bs=256;lr=0.01;epochs =40
    elif network == 'resnet50':
        nsupdate=16
        network = 'resnet50';bs=512;lr=0.01;epochs =35
    global_max_epochs = epochs
    path = LOGHOME+'/%s/%s-n4-bs%d-lr%.4f-ns%d-sg1.50-ds%s' % (prefix, network,bs,lr, nsupdate,density)
    print(network, path)
    plts = []
    if network == 'lstm':
        line = plot_with_params(network, 4, 100, 30.0, HOSTNAME, r'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, 100, 30.0, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'resnet20':
        line = plot_with_params(network, 4, 32, lr, HOSTNAME, 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, bs, lr, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
        pass
    elif network == 'vgg16':
        line = plot_with_params(network, 4, bs, lr, HOSTNAME, 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, bs, lr, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'lstman4':
        line = plot_with_params(network, 4, 8, 0.0002, HOSTNAME, 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, 8, 0.0002, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'resnet50':
        line = plot_with_params(network, 4, 512, lr, HOSTNAME, 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=nsupdate, force_legend=True)
        line = plot_with_params(network, 4, 512, lr, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=nsupdate, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'alexnet':
        plot_with_params(network, 4, 256, lr, HOSTNAME, 'S-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 256, lr, HOSTNAME, r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=nsupdate, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    arr = []
    arr2 = []
    for i in range(1, epochs+1):
        fn = '%s/gtopknorm-rank0-epoch%d.npy' % (path, i)
        fn2 = '%s/randknorm-rank0-epoch%d.npy' % (path, i)
        arr.append(np.mean(np.power(np.load(fn), 2)))
        arr2.append(np.mean(np.power(np.load(fn2), 2)))
    arr = np.array(arr)
    arr2 = np.array(arr2)
    cax = lax if lax is not None else ax1
    cax.plot(arr/arr2, label=r'$\delta$', color=fixed_colors['blue'],linewidth=1)
    cax.set_ylim(bottom=0.97, top=1.001)
    zero_x = np.arange(len(arr), step=1)
    ones = np.ones_like(zero_x)
    cax.plot(zero_x, ones, ':', label='1 ref.', color='black', linewidth=1)
    if True or network.find('lstm') >= 0:
        subaxes = inset_axes(cax,
                            width='50%', 
                            height='30%', 
                            bbox_to_anchor=(-0.04,0,1,0.95),
                            bbox_transform=cax.transAxes,
                            loc='upper right')
        half = epochs //2
        subx = np.arange(half, len(arr))
        subaxes.plot(subx, (arr/arr2)[half:], color=fixed_colors['blue'], linewidth=1)
        subaxes.plot(subx, ones[half:], ':', color='black', linewidth=1)
        subaxes.set_ylim(bottom=subaxes.get_ylim()[0])
    cax.set_xlabel('# of iteration')
    cax.set_ylabel(r'$\delta$')
    u.update_fontsize(cax, FONTSIZE)
    if global_index is not None:
        global_index += 1
    return plts


def plot_group_lr_sensitivies():
    def _plot_with_network(network):
        global global_max_epochs
        global global_density
        densities = [0.001, 0.00025, 0.0001, 0.00005]
        if network == 'vgg16':
            global_max_epochs = 140
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 128, 0.1, HOSTNAME, legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        elif network == 'resnet20':
            global_max_epochs = 140
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 32, 0.1, HOSTNAME, legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        elif network == 'lstm':
            global_max_epochs = 40
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 100, 30.0, HOSTNAME, legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
        elif network == 'lstman4':
            global_max_epochs = 80
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 8, 0.0002, HOSTNAME, legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai2019', nsupdate=1, sg=1.5, density=density, force_legend=True)
    global ax
    networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        ax = group_axs[ax_row][ax_col]
        _plot_with_network(network)
        ax.legend(ncol=2, loc='upper right', fontsize=FONTSIZE-2)
    plt.subplots_adjust(bottom=0.10, left=0.10, right=0.94, top=0.95, wspace=0.37, hspace=0.42)
    plt.savefig('%s/multiple_lrs.pdf'%OUTPUTPATH)


if __name__ == '__main__':
    if PLOT_NORM:
        plot_group_norm_diff()
    else:
        plot_group_lr_sensitivies()
    plt.show()
