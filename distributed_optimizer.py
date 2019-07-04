from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import threading
import allreducer as ar
import torch
import torch.nn as nn
import time
from mpi4py import MPI
from compression import NoneCompressor
from settings import logger
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue
else:
    import queue as Queue


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compressor, is_sparse=True, err_handler=None, layerwise_times=None, sigma_scale=2.5, density=0.01, norm_clip=None, writer=None):
        super(self.__class__, self).__init__(params)
        self._compressor= compressor 
        self._sparse = is_sparse 
        self._layerwise_times = layerwise_times
        self._msg_queue = Queue.Queue()
        self._msg_queue2 = Queue.Queue()

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._register_hooks()

        self._lock = threading.Lock()
        self._key_lock = threading.Lock()
        self.momentum_correction = False
        self._allreducer = ar.AllReducer(named_parameters, self._lock, self._key_lock, compressor, sparse=self._sparse, err_callback=err_handler, layerwise_times=layerwise_times, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, msg_queue=self._msg_queue, msg_queue2=self._msg_queue2, writer=writer)
        self.allreducer_thread = threading.Thread(name='allreducer', target=self._allreducer.run)
        self.allreducer_thread.start()
        self.local = False
        self._synced = False

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                name = self._parameter_names.get(p)
                d_p = p.grad.data
                if self.momentum_correction:
                    param_state = self.state[p]
                    momentum = 0.9
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf
                self._handles[p] = self._allreducer.add_tensor(name, d_p)
                torch.cuda.synchronize()
                #if rank() == 0:
                #    logger.info('-->pushed time [%s]: %s, norm: %f', name, time.time(), p.grad.data.norm())
                self._msg_queue.put(name)
        return hook
    
    def synchronize(self):
        if not self._synced:
            self._msg_queue2.get() # wait for allreducer 
            self._synced = True
        for p, value in self._handles.items():
            output = self._allreducer.get_result(value)
            p.grad.data.set_(output.data)
        self._handles.clear()

    def _step(self, closure=None):
        """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                name = self._parameter_names.get(p)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                #if name.find('bias') >= 0 or name.find('bn') >= 0:
                #    print('batch norm or bias detected, continue, %s' % name)
                if momentum != 0 and not self.momentum_correction:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss

    def _step_with_mc(self, closure=None):
        """Performs a single optimization step with momemtum correction.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                name = self._parameter_names.get(p)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss

    def step(self, closure=None):
        if not self.local:
            self.synchronize()
        ret = self._step(closure)
        self._synced = False
        return ret

    def stop(self):
        self._allreducer.stop()
        self._msg_queue.put('STOP')

    def add_train_epoch(self):
        self._allreducer.train_epoch += 1

    def get_current_density(self):
        return self._allreducer.get_current_density()


def DistributedOptimizer(optimizer, named_parameters=None, compression=NoneCompressor, is_sparse=False, err_handler=None, layerwise_times=None, sigma_scale=2.5, density=0.1, norm_clip=None, writer=None):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression, is_sparse, err_handler, layerwise_times, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer)

def rank():
    return MPI.COMM_WORLD.rank

def size(self):
    return MPI.COMM_WORLD.size

