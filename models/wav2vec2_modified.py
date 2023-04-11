import torch
import torch.nn as nn

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class WAV2VEC2_MODIFIED(Wav2Vec2PreTrainedModel):
    def __init__(self, config=None, args=None):
        super().__init__(args)
        self.num_labels = args.num_labels
        self.pooling_mode = args.pooling_mode
        self.args = args
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        
        self.init_weights()
    
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, x, attention_mask=None):
        print(x)


        # outputs = self.wav2vec2(x, attention_mask=attention_mask)

        print(x)
        exit(0)