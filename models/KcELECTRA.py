import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification

class KCELECTRA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrained_model = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_labels=7, problem_type="multi_label_classification")
    
    def forward(self, x1, x2):
        # return F.sigmoid(self.pretrained_model(x1, attention_mask=x2)[0])
        output = self.pretrained_model(x1, attention_mask=x2)[0]
        return output

    def train(self):
        self.pretrained_model.train()
    
    def eval(self):
        self.pretrained_model.eval()
