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

        # self.ff1 = nn.Linear(768, 1024)
        # self.ff2 = nn.Linear(1024, 1024)
        # self.ff3 = nn.Linear(1024, 512)
        # self.ff4 = nn.Linear(512, 256)
        # self.ff5 = nn.Linear(256, self.num_labels)

        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(1024)
        # self.bn3 = nn.BatchNorm1d(512)
        # self.bn4 = nn.BatchNorm1d(256)

        # self.dp1 = nn.Dropout(0.1)
        # self.dp2 = nn.Dropout(0.1)
        # self.dp3 = nn.Dropout(0.1)
        # self.dp4 = nn.Dropout(0.1)

        self.dense = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, self.num_labels)
    
    def forward(self, x):
        # output = self.dp1(self.bn1(F.relu(self.ff1(x1))))
        # output = self.dp2(self.bn2(F.relu(self.ff2(output))))
        # output = self.dp3(self.bn3(F.relu(self.ff3(output))))
        # output = self.dp4(self.bn4(F.relu(self.ff4(output))))

        # output = self.ff5(output)
        # return output

        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
