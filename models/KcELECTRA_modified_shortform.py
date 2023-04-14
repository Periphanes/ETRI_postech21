import torch.nn as nn
import torch.nn.functional as F


class KCELECTRA_MODIFIED_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_labels = args.num_labels

        self.dense = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, self.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
