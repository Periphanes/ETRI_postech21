import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel

class KCELECTRA_MODIFIED_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_labels = args.num_labels

        self.ff1 = nn.Linear(1536, 1024)
        self.ff2 = nn.Linear(1024, 1024)
        self.ff3 = nn.Linear(1024, self.num_labels)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
    
    def forward(self, x1):
        output = self.dp1(self.bn1(F.relu(self.ff1(x1))))
        output = self.dp2(self.bn2(F.relu(self.ff2(output))))
        output = self.ff3(output)
        return output
