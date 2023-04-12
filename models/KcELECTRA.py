import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel

class KCELECTRA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.txt_feature_extractor = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=7, problem_type="multi_label_classification")
        self.txt_feature_extractor.resize_token_embeddings(54349)
    
    def forward(self, x1, x2):
        return self.txt_feature_extractor(x1, attention_mask=x2)[0]
