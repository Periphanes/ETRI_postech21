import torch
import torch.nn as nn
from transformers.models.mobilebert.modeling_mobilebert import Bottleneck
import torch.nn.functional as F

def apply_xavier(layer):
    if hasattr(layer, 'weight'):
        print(layer)
        torch.nn.init.xavier_uniform_(layer.weight)

class MultimodalBottleneckTransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.transformer_ff_dim = args.transformer_ff_dim
        self.transformer_dropout = args.transformer_dropout
        self.transformer_activation = args.transformer_activation
        self.bottleneck_length = args.bottleneck_length
        self.transformer_num_head = args.transformer_heads
        self.audio_weight = args.audio_weight
        self.txt_weight = args.txt_weight

        self.audio_transformer_layer = nn.TransformerEncoderLayer(512 + self.bottleneck_length, nhead=self.transformer_num_head, dim_feedforward=self.transformer_ff_dim, dropout=self.transformer_dropout, activation=self.transformer_activation)
        self.txt_transformer_layer = nn.TransformerEncoderLayer(512 + self.bottleneck_length, nhead=self.transformer_num_head, dim_feedforward=self.transformer_ff_dim, dropout=self.transformer_dropout, activation=self.transformer_activation)

        apply_xavier(self.audio_transformer_layer)
        apply_xavier(self.txt_transformer_layer)

    def forward(self, x):
        x_audio, x_bottle, x_txt = x[0], x[1], x[2]

        audio_bot = torch.cat((x_bottle, x_audio), dim=1)           # (16, 256 + 512)
        txt_bot = torch.cat((x_bottle, x_txt), dim=1)               # (16, 256 + 512)

        audio_out = self.audio_transformer_layer(audio_bot)         # (16, 256 + 512)
        txt_out = self.txt_transformer_layer(txt_bot)               # (16, 256 + 512)

        bot_audio_part = audio_out[:, :self.bottleneck_length]             # (16, 256)
        bot_txt_part = txt_out[:, :self.bottleneck_length]                 # (16, 256)

        bot_audio_part = torch.mul(bot_audio_part, self.audio_weight)
        bot_txt_part = torch.mul(bot_txt_part, self.txt_weight)

        x_bottle = torch.div(torch.add(bot_audio_part, bot_txt_part), self.audio_weight + self.txt_weight)    # (16, 256)

        x_audio = audio_out[:, self.bottleneck_length:]                                # (16, 512)
        x_txt = txt_out[:, self.bottleneck_length:]                                    # (16, 512)

        return (x_audio, x_bottle, x_txt)

class AUDIO_TXT_ASYMMETRICAL_MBT_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.txt_resize_ff = nn.Linear(768, 512)
        # self.audio_resize_ff = nn.Linear(512, 512)
        self.audio_resize_ff = nn.Linear(1024, 512)

        self.bottleneck_tokens = nn.Parameter(torch.randn(args.batch_size, args.bottleneck_length)).to(args.device)

        self.transformer_num_layers = args.transformer_layers
        self.batch_size = args.batch_size

        self.mbt_layers = nn.ModuleList()
        for i in range(self.transformer_num_layers):
            self.mbt_layers.append(MultimodalBottleneckTransformerLayer(args))

        # self.final_layer = nn.Linear(512*2, args.num_labels)
        self.final_layer = nn.Linear(args.bottleneck_length, args.num_labels)

    def forward(self, audio_out, txt_out):

        txt_out = self.txt_resize_ff(txt_out)
        audio_out = self.audio_resize_ff(audio_out)
        txt_out = F.gelu(txt_out)
        audio_out = F.gelu(audio_out)

        mbt_out = (audio_out, self.bottleneck_tokens, txt_out)              # ((16, 512), (16, 256), (16, 512))

        for i in range(self.transformer_num_layers):
            mbt_out = self.mbt_layers[i](mbt_out)                           # ((16, 512), (16, 256), (16, 512))

        # out = torch.cat((mbt_out[0], mbt_out[2]), dim=1)
        out = mbt_out[1]
        out = self.final_layer(out)

        return out
