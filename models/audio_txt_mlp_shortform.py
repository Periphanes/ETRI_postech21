import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AUDIO_TXT_MLP_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_labels = args.num_labels
        self.batch_size = args.batch_size

        self.ff1 = nn.Linear(768 + 512, 1024)
        self.ff2 = nn.Linear(1024, 1024)
        self.ff3 = nn.Linear(1024, self.num_labels)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

    
    def forward(self, audio_out, txt_out):
        output = torch.cat([audio_out, txt_out], dim=1)

        output = self.dp1(self.bn1(F.relu(self.ff1(output))))
        output = self.dp2(self.bn2(F.relu(self.ff2(output))))
        output = self.ff3(output)

        return output