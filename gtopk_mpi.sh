#!/bin/bash
dnn="${dnn:-resnet20}"
density="${density:-0.001}"
source exp_configs/$dnn.conf
compressor="${compressor:-gtopk}"
nworkers="${nworkers:-4}"
nwpernode=4
sigmascale=2.5
PY=python
mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers --bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    $PY -m mpi4py gtopk_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor
