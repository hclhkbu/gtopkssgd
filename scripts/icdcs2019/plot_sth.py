from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import os 
import sys
import argparse
import math
from utils import read_log, plot_hist, update_fontsize

OUTPUT_PATH = '/media/sf_Shared_Data/tmp/infocom19'

comp_color = '#2e75b6'
compression_color = 'g'
comm_color = '#c00000'
synceasgd_color = '#3F5EBA'
opt_comm_color = '#c55a11'


class Bar:
    initialized = False
    def __init__(self, start_time, duration, max_time, ax, type='p', index=1, is_optimal=False):
        """
        type: p for compute, m for communication
        """
        self.start_time_ = start_time
        self.ax_ = ax
        self.max_time_ = max_time
        self.duration_ = duration/max_time
        self.type_ = type
        self.height_ = 0.1
        self.start_ = 0.3
        self.index_ = index
        self.is_optimal_ = is_optimal
        self.y_ = self.start_+self.height_ if self.type_ is 'p' else self.start_
        if self.type_ == 'p':
            self.y_ = self.start_+self.height_
            self.color_ = comp_color
        elif self.type_ == 'wc': # WFBP
            self.y_ = self.start_
            self.color_ = comm_color
        elif self.type_ == 'sc': # SyncEASGD
            self.y_ = self.start_-self.height_
            self.color_ = synceasgd_color
        elif self.type_ == 'mc': # MG-WFGP
            self.y_ = self.start_-2*self.height_-self.height_*0.2
            self.color_ = opt_comm_color 
        #self.color_ = comp_color if self.type_ is 'p' else comm_color
        #if self.is_optimal_:
            #self.y_ = self.start_-self.height_ - self.height_ * 0.2
            #self.color_ = opt_comm_color 
        if not Bar.initialized:
            self.ax_.set_xlim(right=1.05)
            self.ax_.spines['top'].set_visible(False)
            self.ax_.spines['right'].set_visible(False)
            bottom = 0.00#self.start_-self.height_/2
            self.ax_.set_ylim(bottom=bottom, top=0.5)
            self.ax_.get_yaxis().set_ticks([])
            self.ax_.xaxis.set_ticks_position('bottom')
            self.ax_.set_xticks([i*0.1 for i in range(0,11)])
            self.ax_.set_xticklabels([str(int(self.max_time_*(i*0.1)/1000)) for i in range(0,11)])
            self.ax_.arrow(0, 0.05, 1.01, 0., fc='k', ec='k', lw=0.1, color='black', length_includes_head= True, clip_on = False, overhang=0, width=0.0004)
            self.ax_.annotate(r'$t$ $(ms)$', (1.015, 0.07), color='black',  
                                    fontsize=20, ha='center', va='center')
            fontsize = 18 
            left_margin = 0.
            self.ax_.text(left_margin, self.start_+3*self.height_/2, 'Comp.',horizontalalignment='right', color='black', va='center', size=fontsize)
            self.ax_.text(left_margin, self.start_+self.height_/2, 'Comm.(WF.)',horizontalalignment='right', color='black', va='center',size=fontsize)
            self.ax_.text(left_margin, self.start_-self.height_/2, 'Comm.(S.E.)',horizontalalignment='right', color='black', va='center',size=fontsize)
            self.ax_.text(left_margin, self.start_-self.height_-self.height_/2, 'Comm.(M.W.)',horizontalalignment='right', color='black', va='center',size=fontsize)
        Bar.initialized = True

    def render(self):
        x = self.start_time_ / self.max_time_
        y = self.y_
        if self.duration_ > 0.0:
            rect =  Rectangle((x, y), self.duration_, self.height_, axes=self.ax_, color=self.color_, ec='black', alpha=0.8)
            self.ax_.add_patch(rect)
            fz = 16
            if str(self.index_).find(',') > 0:
                fz = 16
                self.index_ = self.index_.split(',')[0]+'-'+self.index_.split(',')[-1]
            self.ax_.annotate(str(self.index_), (x, y+0.02), color='black',  
                                    fontsize=fz, ha='left', va='center')
        #return rect

def render_log(filename):
    sizes = []
    computes = []
    comms = []
    sizes, comms, computes, merged_comms = read_log(filename)
    #sizes = sizes[::-1]
    #computes = computes[::-1]
    #comms = comms[::-1]
    start_time = 0.0
    comm_start_time = 0.0
    comm = 0.0
    max_time = max(np.sum(computes), np.sum(comms)+computes[0])
    fig, ax = plt.subplots(1)
    print('sizes: ', sizes)
    print('computes: ', computes)
    print('communications: ', comms)
    for i in range(len(computes)):
        comp = computes[i]
        bar = Bar(start_time, comp, max_time, ax, type='p')
        bar.render()
        if comm_start_time + comm > start_time + comp:
            comm_start_time = comm_start_time + comm
        else:
            comm_start_time = start_time + comp
        comm = comms[i]
        bar_m = Bar(comm_start_time, comm, max_time, ax, type='m')
        bar_m.render()
        start_time += comp 
    plt.show()
    plt.clf()
    plt.scatter(sizes, comms, c='blue')
    plt.scatter(sizes, computes, c='red')
    plt.show()

def allreduce_log(filename):
    f = open(filename, 'r')
    num_of_nodes = 2 
    sizes = []
    comms = []
    for l in f.readlines():
        if l[0] == '#' or len(l)<10:
            continue
        items = ' '.join(l.split()).split()
        comm = float(items[-1])
        size = int(items[0].split(',')[1])
        num_of_nodes = int(items[0].split(',')[0])
        comms.append(comm)
        sizes.append(size)
    f.close()
    #print('num_of_nodes: ', num_of_nodes)
    #print('sizes: ', sizes)
    #print('comms: ', comms)
    return num_of_nodes, sizes, comms


def plot_allreduce_log(filenames):
    markers=['-ro', '-go', '-bo']
    for index, fn in enumerate(filenames):
        num_of_nodes, sizes, comms = allreduce_log(fn)
        line1, = plt.plot(sizes, comms, markers[index])
    plt.show()
    plt.clf()

ax = None
def statastic_gradient_size(filename, label, color, marker):
    global ax
    sizes, comms, computes, merged_comms = read_log(filename)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4.5))
    fontsize = 14 
    ax.scatter(range(1, len(sizes)+1), sizes, c=color, label=label, marker=marker, s=40, facecolors='none', edgecolors=color)
    #plot_hist(sizes)
    ax.set_xlim(left=0)
    ax.set_xlabel('Learnable layer ID')
    #plt.ylim(bottom=1e3, top=1e7)
    #plt.ylabel('Message size (bytes)')
    ax.set_ylabel('# of parameters')
    ax.set_yscale("log", nonposy='clip')
    ax.legend()
    update_fontsize(ax, fontsize)
    print('total size: ', np.sum(sizes))
    return sizes

def statastic_gradient_size_all_cnns():
    filenames = []
    for nn in ['googlenet', 'resnet', 'densenet']:
        f = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % nn
        filenames.append(f)
    #cnns = ['GoogleNet', 'ResNet-50', 'DenseNet']
    cnns = ['GoogleNet', 'ResNet-50']
    colors = ['r', 'g', 'b']
    markers = ['+', 'x', 'd']
    sizes = []
    for i, f in enumerate(filenames):
        if i >= len(cnns):
            break
        s = statastic_gradient_size(f, cnns[i], colors[i], markers[i])
        sizes.extend(s)
    #plt.subplots_adjust(left=0.16, bottom=0.13, top=0.93, right=0.96)
    plt.subplots_adjust(left=0.18, bottom=0.13, top=0.91, right=0.92)
    plt.show()
    #plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'gradient_distribution'))
    sizes = np.array(sizes)
    print(np.max(sizes), np.min(sizes))


if __name__ == '__main__':
    #render_log(test_file)
    #statastic_gradient_size(test_file)
    statastic_gradient_size_all_cnns()
