import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraForSequenceClassification
from transformers import ElectraModel, AutoConfig

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2FeatureExtractor(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode

        self.wav2vec2 = Wav2Vec2Model(config)
        self.args = {}
        
        self.init_weights()
    
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def forward(self, x, attention_mask=None):
        outputs = self.wav2vec2(x, attention_mask=attention_mask)
        return outputs['extract_features']

class MultimodalBottleneckTransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__(args)

        transformer_num_head = 4
        transformer_ff_dim = 2048

        self.audio_transformer_layer = nn.TransformerEncoderLayer(512, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)
        self.txt_transformer_layer = nn.TransformerEncoderLayer(512, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)

    def forward(self, x):
        x_audio, x_bottle, x_txt = x[0], x[1], x[2]

        audio_bot = torch.cat((x_bottle, x_audio), dim=1)
        txt_bot = torch.cat((x_bottle, x_txt), dim=1)

        audio_out = self.audio_transformer_layer(audio_bot)
        txt_out = self.txt_transformer_layer(txt_bot)

        bot_audio_part = audio_out[:,:x_bottle.shape[1],:]
        bot_txt_part = txt_out[:,:x_bottle.shape[1],:]
        x_bottle = torch.div(torch.add(bot_audio_part, bot_txt_part), 2)

        x_audio = audio_out[:,x_bottle.shape[1]:,:]
        x_txt = txt_out[:,x_bottle.shape[1]:,:]

        return (x_audio, x_bottle, x_txt)

class AUDIO_TXT_MBT(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        config = AutoConfig.from_pretrained(
            "kresnik/wav2vec2-large-xlsr-korean",
            num_labels = args.num_labels,
            finetuning_task = "wav2vec2_clf"
        )
        setattr(config, 'pooling_mode', args.pooling_mode)

        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean", config=config)
        self.txt_feature_extractor = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.txt_feature_extractor.resize_token_embeddings(54349)

        self.txt_resize_ff = nn.Linear(768, 512)

        self.bottleneck_tokens = nn.Parameter(torch.randn(args.batch_size, args.bottleneck_length, 512)).to(args.device)
        self.audio_cls_tokens = nn.Parameter(torch.randn(args.batch_size, 1, 512)).to(args.device)
        self.txt_cls_tokens = nn.Parameter(torch.randn(args.batch_size, 1, 512)).to(args.device)

        transformer_num_layers = 4
        self.transformer_num_layers = transformer_num_layers

        self.mbt_layers = nn.ModuleList()
        for i in range(transformer_num_layers):
            self.mbt_layers.append(MultimodalBottleneckTransformerLayer(args))

        self.ff_txt1 = nn.Linear(512, 64)
        self.ff_txt2 = nn.Linear(64,args.num_labels)
        self.ff_audio1 = nn.Linear(512, 64)
        self.ff_audio2 = nn.Linear(64,args.num_labels)

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x_audio, x_audio_attn, x_txt, x_txt_attn):
        audio_out = self.audio_feature_extractor(x_audio, attention_mask=x_audio_attn)
        txt_out = self.txt_feature_extractor(x_txt, attention_mask=x_txt_attn).last_hidden_state
        txt_out = self.txt_resize_ff(txt_out)

        audio_out = torch.cat((self.audio_cls_tokens, audio_out), dim=1)
        txt_out = torch.cat((self.txt_cls_tokens, txt_out), dim=1)

        mbt_out = (audio_out, self.bottleneck_tokens, txt_out)

        for i in range(self.transformer_num_layers):
            mbt_out = self.mbt_layers[i](mbt_out)

        cls_audio_out = mbt_out[0]
        audio_out = self.ff_audio1(cls_audio_out)
        audio_out = self.sigmoid(self.ff_audio2(audio_out))

        cls_txt_out = mbt_out[1]
        txt_out = self.ff_txt1(cls_txt_out)
        txt_out = self.sigmoid(self.ff_txt2(txt_out))

        out = torch.div(torch.add(txt_out, audio_out), 2)

        return out