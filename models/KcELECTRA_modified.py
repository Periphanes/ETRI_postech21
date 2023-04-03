import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel

class KCELECTRA_MODIFIED(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.pretrained_model = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_labels=7, problem_type="multi_label_classification")
        self.pretrained_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base")
        
        self.ff1 = nn.Linear(768, 768)
        self.ff2 = nn.Linear(768, 256)
        self.ff3 = nn.Linear(256, 7)
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(256)
    
    def forward(self, x1, x2):
        # return F.sigmoid(self.pretrained_model(x1, attention_mask=x2)[0])
        output = self.pretrained_model(x1, attention_mask=x2).last_hidden_state
        output = output[:, 0, :]
        output = F.relu(self.bn1(self.ff1(output)))
        output = F.relu(self.bn2(self.ff2(output)))
        output = self.ff3(output)
        return output
