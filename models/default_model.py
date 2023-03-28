import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DEFAULT_MODEL(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
    
    def forward(self, x):

        return x