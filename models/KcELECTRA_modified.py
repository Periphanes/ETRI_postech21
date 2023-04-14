import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel

class KCELECTRA_MODIFIED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.txt_feature_extractor = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.txt_feature_extractor.resize_token_embeddings(54349)
        
        self.num_labels = args.num_labels

        # self.ff1 = nn.Linear(768, 1024)
        # self.ff2 = nn.Linear(1024, 1024)
        # self.ff3 = nn.Linear(1024, self.num_labels)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(1024)
        # self.dp1 = nn.Dropout(0.1)
        # self.dp2 = nn.Dropout(0.1)

        self.dense = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, self.num_labels)
    
    def forward(self, x1, x2):
        # output = self.txt_feature_extractor(x1, attention_mask=x2).last_hidden_state
        # output = output[:, 0, :]
        # output = self.dp1(self.bn1(F.relu(self.ff1(output))))
        # output = self.dp2(self.bn2(F.relu(self.ff2(output))))
        # output = self.ff3(output)
        # return output

        features = self.txt_feature_extractor(x1, attention_mask=x2).last_hidden_state
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
