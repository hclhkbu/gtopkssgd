#python evaluate.py --model-path weights/allreduce/resnet20-n4-bs32-lr0.1000 --dnn resnet20 --dataset cifar10 --nepochs 140 --data-dir ./data
#python evaluate.py --model-path weights/comp-gtopk-baseline-gwarmup-dc1-model-debug2-ds0.001/resnet110-n4-bs128-lr0.1000 --dnn resnet110 --dataset cifar10 --nepochs 140 --data-dir ./data
python evaluate.py --model-path weights/comp-gtopk-baseline-gwarmup-dc1-model-debug2-ds0.001/resnet110-n4-bs128-lr0.1000 --dnn resnet110 --dataset cifar10 --nepochs 140 --data-dir ./data
#python evaluate.py --model-path weights/comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1-ds0.001/resnet20-n4-bs32-lr0.1000 --dnn resnet20 --dataset cifar10 --nepochs 140 --data-dir ./data
#python evaluate.py --model-path weights/allreduce/vgg16-n4-bs128-lr0.1000 --dnn vgg16 --dataset cifar10 --nepochs 140 --data-dir ./data
#python evaluate.py --model-path weights/comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1-ds0.001/vgg16-n4-bs128-lr0.1000 --dnn vgg16 --dataset cifar10 --nepochs 140 --data-dir ./data

#python evaluate.py --model-path weights/comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1-ds0.001/lstm-n4-bs32-lr1.0000 --dnn lstm --dataset ptb --nepochs 40 --data-dir /home/shshi/data/PennTreeBank
#python evaluate.py --model-path weights/allreduce/lstm-n4-bs5-lr20.0000 --dnn lstm --dataset ptb --nepochs 40 --data-dir /home/shshi/data/PennTreeBank
#python evaluate.py --model-path weights/allreduce/lstm-n4-bs5-lr20.0000 --dnn lstm --dataset ptb --nepochs 40 --data-dir /home/comp/csshshi/data/PennTreeBank

#python evaluate.py --model-path weights/allreduce/lstman4-n4-bs8-lr0.0003 --dnn lstman4 --dataset an4 --nepochs 90 --data-dir /home/comp/csshshi/data/an4data
#python evaluate.py --model-path weights/allreduce/lstman4-n4-bs32-lr0.0003 --dnn lstman4 --dataset an4 --nepochs 90 --data-dir /home/comp/csshshi/data/an4data
