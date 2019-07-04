#kill -9 `ps aux|grep 'python client.py' | awk '{print $2}'`
#kill -9 `ps aux|grep 'python client_mp.py' | awk '{print $2}'`
#kill -9 `ps aux|grep 'python psclient.py' | awk '{print $2}'`
#kill -9 `ps aux|grep 'python dl_trainer.py' | awk '{print $2}'`
kill -9 `ps aux|grep 'python -m mpi4py robust_trainer.py' | awk '{print $2}'`
#kill -9 `ps aux|grep 'python robust_trainer.py' | awk '{print $2}'`
kill -9 `ps aux|grep 'python horovod_trainer.py' | awk '{print $2}'`
kill -9 `ps aux|grep 'python -m mpi4py horovod_trainer.py' | awk '{print $2}'`
kill -9 `ps aux|grep 'python -m mpi4py hovorod_trainer.py' | awk '{print $2}'`
