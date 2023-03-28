import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from control.config import args

def binary_classification_static(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x)
        output = output.squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step(iteration)

    else:
        output = model(x)
        output = output.squeeze()

        loss = criterion(output, y)
    
    return model, loss.item()