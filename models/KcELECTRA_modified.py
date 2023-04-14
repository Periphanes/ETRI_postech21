import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraModel


class KCELECTRA_MODIFIED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.txt_feature_extractor = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.txt_feature_extractor.resize_token_embeddings(54349)

        self.num_labels = args.num_labels

        self.dense = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, self.num_labels)

    def forward(self, x1, x2):
        features = self.txt_feature_extractor(x1, attention_mask=x2).last_hidden_state
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
