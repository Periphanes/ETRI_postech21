import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel

class KCELECTRA_MODIFIED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrained_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.pretrained_model.resize_token_embeddings(54349)
        
        self.ff1 = nn.Linear(768, 1024)
        self.ff2 = nn.Linear(1024, 1024)
        self.ff3 = nn.Linear(1024, 7)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

        self.ff_test = nn.Linear(768, 512)
    
    def forward(self, x1, x2):
        # output = self.pretrained_model(x1, attention_mask=x2)
        # print(output.last_hidden_state.shape)

        # k = self.ff_test(output.last_hidden_state)
        # print(k.shape)
        # exit(1)

        output = self.pretrained_model(x1, attention_mask=x2).last_hidden_state
        output = output[:, 0, :]
        output = self.dp1(self.bn1(F.relu(self.ff1(output))))
        output = self.dp2(self.bn2(F.relu(self.ff2(output))))
        output = self.ff3(output)
        return output
