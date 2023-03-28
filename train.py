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

pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")

for epoch in range(1, args.epochs+1):
    training_loss = []
    validation_loss = []

    epoch_losses    = []
    loss            = 0
    iter_in_epoch   = 0

    for train_batch in train_loader:
        if args.trainer == "binary_classification_static":
            train_x, train_y = train_batch
    
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        iteration               += 1
        iter_in_epoch           += 1
        total_epoch_iteration   += 1

        if args.trainer == "binary_classification_static":
            model, iter_loss = get_trainer(args = args,
                                           iteration = iteration,
                                           x = train_x,
                                           static = None,
                                           y = train_y,
                                           model = model,
                                           device = device,
                                           scheduler=scheduler,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           flow_type="train")
        
        training_loss.append(iter_loss)

        # print("Training Loss : {}".format(iter_loss))


        # Validation Step Start
        if iteration % (iter_num_per_epoch) == 0:
            model.eval()
            val_iteration   = 0

            validation_loss = []

            with torch.no_grad():
                for idx, val_batch in enumerate(val_loader):
                    if args.trainer == "binary_classification_static":
                        val_x, val_y = val_batch
                    
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)

                    if args.trainer == "binary_classification_static":
                        model, val_loss = get_trainer(args = args,
                                                      iteration = iteration,
                                                      x = val_x,
                                                      static = None,
                                                      y = val_y,
                                                      model = model,
                                                      device = device,
                                                      scheduler=scheduler,
                                                      optimizer=optimizer,
                                                      criterion=criterion,
                                                      flow_type="test")
                    
                    val_iteration += 1
                    validation_loss.append(val_loss)
            model.train()
    pbar.update(1)

    pbar.set_description("Training Loss : " + str(sum(training_loss)/len(training_loss)) + " / Val Loss : " + str(sum(validation_loss)/len(validation_loss)))
    pbar.refresh()

