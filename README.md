# gTop-*k* S-SGD
## Introduction
This repository contains the codes of the gTop-k S-SGD (Synchronous Schocastic Gradident Descent) papers appeared at *ICDCS 2019* (this version targets at empirical study) and *IJCAI 2019* (this version targets at theorectical study). gTop-k S-SGD is a communication-efficient distributed training algorithm for deep learning. The key idea of gTop-k is that each work only sends/recieves top-k (k could be 0.1% of the gradient dimension d, i.e., k=0.001d) with a tree structure (recursive doubling) so that the communication complexity is O(k logP), where P is the number of workers. The convergence property of gTop-k S-SGD is provable under some weak analytical assumptions. The communication complexity comparision with tranditional ring-based all-reduce (Dense) and Top-k sparsification is shown as follows:

| S-SGD | Complexity | Time Cost  |
| ------------- |:-------------:| -----:|
| Dense | O(d) | 2\alpha(P-1)+2(P-1)/Pd\beta |
| Top-k | O(kP)| \alpha logP+2(P-1)k\beta |
| **gTop-k** | **O(k logP)** |  **\alpha logP+2klogP\beta**   |

For more details about the algorithm, please refer to our papers.

## Installation
### Prerequisites
- Python 2 or 3
- PyTorch-0.4.+
- [OpenMPI-3.1.+](https://www.open-mpi.org/software/ompi/v3.1/)
- [Horovod-0.14.+](https://github.com/horovod/horovod): Optional if not run the dense version
### Quick Start
```
git clone https://github.com/hclhkbu/gtopkssgd.git
cd gtopkssgd
pip install -r requirements.txt
dnn=resnet20 nworkers=4 ./gtopk_mpi.sh
```
Assume that you have 4 GPUs on a single node and everything works well, you will see that there are 4 workers running at a single node training the ResNet-20 model with the Cifar-10 data set using the gTop-k S-SGD algorithm.
## Papers
- S. Shi, Q. Wang, K. Zhao, Z. Tang, Y. Wang, X. Huang, and X.-W. Chu, “A Distributed Synchronous SGD Algorithm with Global Top-k Sparsification for Low Bandwidth Networks,” *IEEE ICDCS 2019*, Dallas, Texas, USA, July 2019. [PDF](https://arxiv.org/pdf/1901.04359.pdf)
- S. Shi, K. Zhao, Q. Wang, Z. Tang, and X.-W. Chu, “A Convergence Analysis of Distributed SGD with Communication-Efficient Gradient Sparsification,”  *IJCAI 2019*, Macau, P.R.C., August 2019. [PDF](https://www.ijcai.org/proceedings/2019/0473.pdf)
## Referred Models
- Deep speech: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
- PyTorch examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
