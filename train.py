# Main Training File

import os
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from control.config import args

log_directory = os.path.join(args.dir_result, args.project_name)

# make sure that CUDA uses GPU according to the inserted order
# not useful unless in multi-GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for k_indx, seed_num in enumerate(args.seed_list):
    args.seed = seed_num
    # set all the seeds in random and other libraries to given seed_num
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setting flags to make sure reproducability for certain seeds
    # cudnn.deterministic weeds out random algorithms,
    # cudnn.benchmark disables benchmarking, which may introduce randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set main device to CPU or GPU(cuda)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    