# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
import torch
import logging
import utils
import settings
from mpi4py import MPI
from settings import logger



class MESSAGE:
    STOP = 'STOP'
    RUNNING = 'RUNNING'

mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16
MPI_TYPES = {
        np.float32: MPI.FLOAT,
        np.float16: mpi_float16
        }

THRESHOLD = 640*1024*1024


def topk_sparse_allreduce(comm, sparse_tensor, storage, indexes=None, dtype=np.float32):
    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.01)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy().astype(np.uint32)
        k = len(indexes)
        values = tensor#[indexes] 

    num_workers = comm.size
    if storage is not None and 'values_1d' in storage:
        values_1d = storage['values_1d']
        indexes_1d = storage['indexes_1d']
        result = storage['result']
    else:
        values_1d = np.zeros(k * num_workers, dtype=np.float32)
        indexes_1d = np.zeros(k * num_workers, dtype=np.uint32)
        result = np.zeros_like(tensor) 
        storage['values_1d'] = values_1d
        storage['indexes_1d'] = indexes_1d
        storage['result'] = result
        
    if dtype != np.float32:
        values_1d = values_1d.astype(dtype)

    result.fill(0)

    if len(indexes) == 0:
        return result, None

    nnz = k
    comm.Allgather(values, values_1d[:num_workers*nnz])
    comm.Allgather(indexes, indexes_1d[:num_workers*nnz])
    return values_1d, indexes_1d, None #result, None


def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:][::-1]
    return indexes, tensor[indexes]

def gtopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor 
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2*k] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                tmp_indexes = recv_values[0:k].astype(np.int)
                tmp_values = recv_values[k:2*k]

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
    else:
        send_values = recv_values[0:2*k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes, indexes, assume_unique=False, return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes # final selected values and indexes


def dense_allreduce(comm, tensor):
    result = np.zeros_like(tensor)
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)
    comm.Barrier()
    return result

def _default_err_callback(new_num_workers, new_rank):
    logger.error('Some process error accurs, number of workers changes to %d, my rank changes to %d', new_num_workers, new_rank)

def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


class AllReducer():
    def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None):
        self._running = False 
        self._msg_queue = msg_queue
        self._msg_queue2 = msg_queue2
        self._writer = writer
        self._profiling = True
        self._entries = {}
        self._keys = []
        self._outputs = {}
        self._residuals = {}
        self._sparse_storages = {}
        self._sparse_storages_topk = {}
        self._sparse = sparse
        self._sigma_scale = sigma_scale
        self._density = density
        self.train_epoch = train_epoch
        self.train_iter = 0
        logger.info('density: %f', self._density)
        self._comm = MPI.COMM_WORLD
        self._comm.Set_errhandler(MPI.ERRORS_RETURN)
        self._layerwise_times = layerwise_times # L->1: Note that the layerwise time is from the last layer to the first
        _named_parameters = list(named_parameters)
        self._named_parameters = {k: v for k, v
                                in _named_parameters}
        self._default_for_reductions = {k: 1 for k, v
                                in _named_parameters}
        self._sequential_keys = [k for k, v in _named_parameters]
        self._lock = lock
        self._key_lock = key_lock
        self._compression = compression
        self._err_callback = err_callback if err_callback else _default_err_callback
        self._norm_clip = norm_clip
        self._generate_merged_parameters()
        self.allocate_sparse_storages()

        self._allreduce_timers = {}
        self._compression_timers = {}
        self._merge_timers = {}
        self._demerge_timers = {}
        self._h2d_times = {}
        self._d2h_times = {}
        self._profiling_norms = []

        #self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
        self._dynamic_densities = [0.004] # the tuned one 
        if self._dynamic_densities is not None:
            self._dynamic_densities.append(self._density)
            logger.info('dynamic densities = %s', self._dynamic_densities)
        self.reset()

        self.allreduce_count = 0

    def _generate_groups_with_threshold(self, threshold):
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                sub_size = 0
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps


    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        groups, key_groupidx_maps = self._generate_groups_with_threshold(THRESHOLD)
        logger.info('groups: %s', groups)
        logger.info('key_groupidx_maps: %s', key_groupidx_maps)
        new_keys = []
        self._merged_parameter_offsets = {}
        for g in groups:
            sub_size = 0
            offsets = []
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
            new_key = ':'.join(g)
            new_keys.append(new_key)
            self._merged_parameters[new_key] = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)
            self._merged_parameter_offsets[new_key] = offsets
        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)
        logger.info('offsets: ', self._merged_parameter_offsets)

    def _push_to_buffer(self, name, tensor):
        if len(self._groups) == len(self._sequential_keys):
            return name, tensor
        group_idx = self._key_groupidx_maps[name]
        g = self._groups[group_idx]
        new_key = ':'.join(g)
        layer_idx = g.index(name)
        offset = self._merged_parameter_offsets[new_key][layer_idx]
        numel = tensor.data.numel()
        self._merged_parameters[new_key].data[offset:offset+numel]= tensor.view(numel).data
        self._groups_flags[group_idx][layer_idx] = 1
        try:
            idx = self._groups_flags[group_idx].index(0)
        except:
            idx = -1
        if idx >= 0:
            return name, None
        return new_key, self._merged_parameters[new_key]

    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            return {name: merged_tensor} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensor = torch.zeros(numel, device=original_tensor.device, dtype=original_tensor.dtype)
            tensor.data = merged_tensor.data[offset:offset+numel]
            tensors[k] = tensor.view(original_tensor.shape)
        return tensors

    def rank(self):
        return self._comm.rank
    
    def size(self):
        return self._comm.size

    def allocate_sparse_storages(self):
        for k, v in self._merged_parameters.items():
            self.allocate_storage(k, v)

    def _print_profiling(self):
        if self._profiling and self.rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(self._allreduce_timers.keys()[0], [])) == 100:
            cts = self._layerwise_times # gpu computation
            mgs = self._merge_timers # merge_times
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            dms = self._demerge_timers# demerge times
            d2hs = self._d2h_times
            h2ds = self._h2d_times
            l = 0
            logger.info('[rank:%d]name[size]: backward, merge, compression, allreduce, demerge, d2h, h2d')
            total_sz = 0
            total_ct = 0.0
            total_mg = 0.0
            total_cp = 0.0
            total_ar = 0.0
            total_dm = 0.0
            total_d2h = 0.0
            total_h2d = 0.0

            for g in self._groups:
                ct = 0.0
                sz = 0
                for k in g:
                    if cts is not None:
                        ct += cts[l]
                    else:
                        ct = 0.0
                    sz += self._sizes[l]
                    total_ct += ct
                    l += 1
                total_sz += sz
                k = ':'.join(g)
                mg = np.mean(mgs[k])
                total_mg += mg
                cp = np.mean(cps[k])
                total_cp += cp
                ar = np.mean(ars[k])
                total_ar += ar
                dm = np.mean(dms[k])
                total_dm += dm
                d2h = np.mean(d2hs.get(k, [0.0]))
                total_d2h += d2h
                h2d = np.mean(h2ds.get(k, [0.]))
                total_h2d += h2d

                logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(), k[0:3]+'...'+k[-3:], sz, ct,mg,cp,ar,dm,d2h,h2d)
                mgs.pop(k, None)
                cps.pop(k, None)
                ars.pop(k, None)
                dms.pop(k, None)
                d2hs.pop(k, None)
                h2ds.pop(k, None)
            logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(), 'total', total_sz, total_ct,total_mg,total_cp,total_ar,total_dm,total_d2h,total_h2d)

    def reset(self):
        self._for_reductions = self._default_for_reductions.copy()
        self._print_profiling()

    def add_tensor(self, name, tensor):
        if name in self._entries:
            return
        self._entries[name] = tensor
        return name

    def get_current_density(self):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        return density

    def get_approximate_sigma_scale(self, density):
        sigma_scale = 1
        if density > 0.7:
            sigma_scale = 0.5
        elif density <= 0.7 and density > 0.05:
            sigma_scale = 1.5
        elif density <= 0.05 and density > 0.01:
            sigma_scale = 2.0
        else:
            sigma_scale = 3.0
        return sigma_scale

    def get_result(self, name):
        return self._outputs[name]

    def allocate_storage(self, name, tensor):
        storage = {}
        self._sparse_storages[name] = storage
        self._sparse_storages_topk[name] = {}
        

    def _sparse_allreduce(self, name, tensor, selected_tensor, original_shape, topk_indexes=None):
        stime = time.time()
        ct = selected_tensor
        if ct.is_cuda: # only transfer the selected k values through PCI-e
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()
        if self._profiling:
            force_insert_item(self._d2h_times, name, time.time()-stime)

        result = None
        included_indexes = None
        full_mean = None
        full_var = None

        if self._compression.name in ['topk', 'topk2']:
            result, global_indexes, included_indexes = topk_sparse_allreduce(self._comm, entry, self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
        elif self._compression.name in ['gtopk']:
            result, global_indexes, included_indexes = gtopk_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)

        r = torch.from_numpy(result)
        gi = torch.from_numpy(global_indexes.astype(np.int64))
        stime = time.time()
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
            final_indexes = gi.cuda(tensor.device, non_blocking=False)
        else:
            final_indexes = gi 

        tensor.fill_(0.0)
        if self._compression.name in ['gtopk']:
            tensor[final_indexes] = r
        elif self._compression.name in ['topk', 'topk2']:
            num_workers = self._comm.size
            nnz = topk_indexes.size(0)
            for i in range(num_workers):
                index = final_indexes[i*nnz:(i+1)*nnz]
                tensor[index] += r[i*nnz:(i+1)*nnz]
            if self._compression.name == 'topk2':
                values, indexes = torch.topk(torch.abs(tensor.data), k=nnz)
                cv, c1, c2 = np.intersect1d(indexes.cpu().numpy(), topk_indexes.cpu().numpy(), assume_unique=False, return_indices=True)
                included_indexes = c2
                values = tensor.data[indexes]
                tensor.data.fill_(0.0)
                tensor.data[indexes] = values.data

        tensor /= self.size()
        if self._profiling:
            force_insert_item(self._h2d_times, name, time.time()-stime)
        return tensor, included_indexes, full_mean

    def _dense_allreduce(self, name, tensor):
        ct = tensor 
        shape = tensor.shape
        if ct.is_cuda:
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()

        result = dense_allreduce(self._comm, entry)

        result = result.reshape(shape)
        r = torch.from_numpy(result)
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
        r /= self.size()
        return r 

    def run(self):
        self._running = True
        logger.info('Allreducer thread started ...')
        while self._running:
            name = self._msg_queue.get()
            if name == 'STOP':
                break
            if name is not None:
                tensor = self._entries[name]

                # Push the tensor to the buffer
                stime = time.time()
                new_name, new_tensor = self._push_to_buffer(name, tensor)
                if self._profiling:
                    force_insert_item(self._merge_timers, new_name, time.time()-stime)

                if new_tensor is None:
                    continue

                # Compress the gradients. It generally drops most of the gradients
                stime = time.time()

                # For comparison purpose ===>
                if settings.PROFILING_NORM:
                    residuals = self._compression.get_residuals(new_name, new_tensor)
                    new_tensor_add_res = residuals.data + new_tensor.data
                    dense_result = self._dense_allreduce(new_name, new_tensor_add_res)
                    dense_std= float(torch.std(dense_result))
                    random_indexes = torch.randperm(dense_result.size(0))
                # For comparison purpose <=== End

                density = self.get_current_density()
                sigma_scale = self.get_approximate_sigma_scale(density)

                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0/self.size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = new_tensor.norm(norm_type)
                    total_norm = param_norm.item() 
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        new_tensor.mul_(clip_coef)

                original_shape = new_tensor.shape
                new_tensor, ctx = self._compression.compress(new_tensor, new_name, sigma_scale=sigma_scale, ratio=density)
                if ctx is not None:
                    selected_tensor = new_tensor[ctx]
                else:
                    selected_tensor = new_tensor
                
                # For comparison purpose ===>
                if settings.PROFILING_NORM:
                    with torch.no_grad():
                        k = ctx.size(0)
                        rand_k = random_indexes[:k]
                        rand_k_tensor = torch.zeros_like(dense_result)
                        rand_k_tensor.data[rand_k] = dense_result.data[rand_k]
                        randk_norm = (dense_result - rand_k_tensor).norm(p=2)
                # For comparison purpose <=== End

                torch.cuda.synchronize()
                if self._profiling:
                    force_insert_item(self._compression_timers, new_name, time.time()-stime)

                # Allreduce on the merged gradients 
                stime = time.time()
                if self._sparse:

                    result, included_indexes, full_mean = self._sparse_allreduce(new_name, new_tensor, selected_tensor, original_shape, topk_indexes=ctx)
                    if included_indexes is not None:
                        if full_mean is not None:
                            self._compression.add_residuals(included_indexes, new_name, full_mean)
                        else:
                            self._compression.add_residuals(included_indexes, new_name)

                    # For comparison purpose ===>
                    if settings.PROFILING_NORM:
                        gtopk_norm = (dense_result - result).norm(p=2)
                        xnorm  = float(dense_result.norm(p=2))
                        upbound = 1.0*(result.size(0)-k)/result.size(0) * xnorm 
                        self._profiling_norms.append((float(gtopk_norm), float(randk_norm), upbound, xnorm, dense_std))
                    # For comparison purpose <=== End
                else:
                    result = self._dense_allreduce(new_name, new_tensor)
                if self._profiling:
                    force_insert_item(self._allreduce_timers, new_name, time.time()-stime)

                # Decouple on the merged gradients 
                stime = time.time()
                tensors = self._pull_from_buffer(new_name, result)
                if self._profiling:
                    force_insert_item(self._demerge_timers, new_name, time.time()-stime)
                for n in tensors:
                    self._outputs[n] = tensors[n] 
                    self._entries.pop(n, None)
                    self._for_reductions.pop(n, None)

            if len(self._for_reductions) == 0:
                self.reset()
                torch.cuda.synchronize()
                self._msg_queue2.put('DONE')
           
    def stop(self):
        self._running = False


def benchmark_gtopk_sparse_allreduce():
    logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #np.random.seed(rank)
    size = 25 * 1024 * 1024
    ratio = 0.001
    tensor = np.random.rand(size).astype(np.float32)
    k = int(tensor.size * ratio)
    indexes, values = utils.topk(tensor, k)
    #indexes, values = topk(tensor, k)
    #logger.info('topk[%d]%s', rank, values)
    tmp = tensor[indexes]
    tensor.fill(0.)
    tensor[indexes] = tmp
    logger.debug('[%d]%s', rank, tensor)
    storage = {}

    t = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    iteration = 10
    stime = time.time()
    for i in range(iteration):
        t,_ = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    total_time = time.time() - stime
    logger.info('average time: %f', total_time/iteration)


if __name__ == '__main__':
    benchmark_gtopk_sparse_allreduce()

