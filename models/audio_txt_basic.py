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


class AUDIO_TXT_BASIC(nn.Module):
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
        self.cls_tokens = nn.Parameter(torch.randn(args.batch_size, 1, 512)).to(args.device)

        transformer_num_head = 4
        transformer_ff_dim = 2048
        transformer_num_layers = 4

        encoder_layer = nn.TransformerEncoderLayer(512, nhead=transformer_num_head, dim_feedforward=transformer_ff_dim)
        self.final_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.ff1 = nn.Linear(512, 64)
        self.ff2 = nn.Linear(64,args.num_labels)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x_audio, x_audio_attn, x_txt, x_txt_attn):
        audio_out = self.audio_feature_extractor(x_audio, attention_mask=x_audio_attn)
        txt_out = self.txt_feature_extractor(x_txt, attention_mask=x_txt_attn).last_hidden_state
        txt_out = self.txt_resize_ff(txt_out)

        concat_out = torch.cat((self.cls_tokens, txt_out, audio_out), dim=1)
        trans_out = self.final_transformer(concat_out)
        cls_out = trans_out[:,0,:]

        out = self.ff1(cls_out)
        out = self.sigmoid(self.ff2(out))

        return out