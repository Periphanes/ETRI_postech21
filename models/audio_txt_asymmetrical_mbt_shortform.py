import torch
import torch.nn as nn
from transformers.models.mobilebert.modeling_mobilebert import Bottleneck


class MultimodalBottleneckTransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        transformer_num_head = 4
        transformer_ff_dim = 1024

        self.audio_transformer_layer = nn.TransformerEncoderLayer(512*2 + 786 * args.bottleneck_length, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)
        self.txt_transformer_layer = nn.TransformerEncoderLayer(512*2 + 786 * args.bottleneck_length, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)

    def forward(self, x):
        x_audio, x_bottle, x_txt = x[0], x[1], x[2]

        audio_bot = torch.cat((x_bottle, x_audio), dim=1)
        txt_bot = torch.cat((x_bottle, x_txt), dim=1)

        audio_out = self.audio_transformer_layer(audio_bot)
        txt_out = self.txt_transformer_layer(txt_bot)

        bot_audio_part = audio_out[:, :x_bottle.shape[1]]
        bot_txt_part = txt_out[:, :x_bottle.shape[1]]

        audio_weight = 5 # weight of audio compared to text
        
        bot_audio_part = torch.mul(bot_audio_part, audio_weight)
        bot_txt_part = torch.mul(bot_txt_part, 1)

        x_bottle = torch.div(torch.add(bot_audio_part, bot_txt_part), audio_weight + 1)

        x_audio = audio_out[:,x_bottle.shape[1]:]
        x_txt = txt_out[:,x_bottle.shape[1]:]

        return [x_audio, x_bottle, x_txt]

class AUDIO_TXT_ASYMMETRICAL_MBT_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.txt_resize_ff = nn.Linear(768, 512)

        self.bottleneck_tokens = nn.Parameter(torch.randn(args.batch_size, 786 * args.bottleneck_length)).to(args.device)
        self.audio_cls_tokens = nn.Parameter(torch.randn(args.batch_size, 512)).to(args.device)
        self.txt_cls_tokens = nn.Parameter(torch.randn(args.batch_size, 512)).to(args.device)

        transformer_num_layers = 4
        self.transformer_num_layers = transformer_num_layers
        self.batch_size = args.batch_size
        self.bottleneck_length = args.bottleneck_length

        self.mbt_layers = nn.ModuleList()
        for i in range(transformer_num_layers):
            self.mbt_layers.append(MultimodalBottleneckTransformerLayer(args))

        self.ff_txt1 = nn.Linear(512 * self.bottleneck_length, args.num_labels)
        self.ff_audio1 = nn.Linear(512 * self.bottleneck_length, args.num_labels)

        self.ln_num_layers = transformer_num_layers
        self.ln_layers1 = nn.ModuleList()
        self.ln_layers2 = nn.ModuleList()
        self.ln_layers3 = nn.ModuleList()
        for i in range(transformer_num_layers):
            self.ln_layers1.append(nn.LayerNorm([512 * self.bottleneck_length]))
            self.ln_layers2.append(nn.LayerNorm([786 * self.bottleneck_length]))
            self.ln_layers3.append(nn.LayerNorm([512 * self.bottleneck_length]))
    
    def forward(self, audio_out, txt_out):
        txt_out = self.txt_resize_ff(txt_out)

        audio_out = torch.cat((self.audio_cls_tokens, audio_out), dim=1)
        txt_out = torch.cat((self.txt_cls_tokens, txt_out), dim=1)

        mbt_out = [audio_out, self.bottleneck_tokens, txt_out]
        
        for i in range(self.transformer_num_layers):
            mbt_out = self.mbt_layers[i](mbt_out)
            mbt_out[0] = self.ln_layers1[i](mbt_out[0])
            mbt_out[1] = self.ln_layers2[i](mbt_out[1])
            mbt_out[2] = self.ln_layers3[i](mbt_out[2])

        cls_audio_out = mbt_out[0][:, :self.bottleneck_length * 512]
        audio_out = self.ff_audio1(cls_audio_out)

        cls_txt_out = mbt_out[2][:, :self.bottleneck_length * 512]
        txt_out = self.ff_txt1(cls_txt_out)

        out = torch.div(torch.add(txt_out, audio_out), 2)

        return out