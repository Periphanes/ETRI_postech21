import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AUDIO_TXT_BASIC_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.txt_resize_ff = nn.Linear(768, 512)
        self.cls_tokens = nn.Parameter(torch.randn(args.batch_size, 1, 512)).to(args.device)

        transformer_num_head = 4
        transformer_ff_dim = 2048
        transformer_num_layers = 4

        encoder_layer = nn.TransformerEncoderLayer(512, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)
        self.final_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.ff1 = nn.Linear(512, 64)
        self.ff2 = nn.Linear(64,args.num_labels)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, audio_out, txt_out):
        txt_out = self.txt_resize_ff(txt_out)

        concat_out = torch.cat((self.cls_tokens, txt_out, audio_out), dim=1)
        trans_out = self.final_transformer(concat_out)
        cls_out = trans_out[:,0,:]

        out = self.ff1(cls_out)
        out = self.ff2(out)

        return out