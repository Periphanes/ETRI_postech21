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
from models import get_model
from trainer import get_trainer
from data.data_preprocess import get_data_loader

log_directory = os.path.join(args.dir_result, args.project_name)

# make sure that CUDA uses GPU according to the inserted order
# not useful unless in multi-GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args.seed = args.seed_list[0]
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

if args.input_types == "static":
    args.trainer = "binary_classification_static"
else:
    raise NotImplementedError("Trainer Not Implemented Yet")

train_loader, val_loader, test_loader = get_data_loader(args)

model = get_model(args)
model = model(args).to(device)

criterion = nn.BCELoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=args.lr_init)

iter_num_per_epoch = len(train_loader)
iter_num_total = args.epochs * iter_num_per_epoch

print("# of Iterations (per epoch): ",  iter_num_per_epoch)
print("# of Iterations (total): ",      iter_num_total)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

model.train()
iteration = 0
total_epoch_iteration = 0

