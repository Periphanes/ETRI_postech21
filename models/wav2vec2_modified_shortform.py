import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers.models.wav2vec2.modeling_wav2vec2 import (
#     Wav2Vec2PreTrainedModel,
#     Wav2Vec2Model
# )

# from transformers import AutoConfig

# class Wav2Vec2FeatureExtractor(Wav2Vec2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
        
#         self.num_labels = config.num_labels
#         self.pooling_mode = config.pooling_mode

#         self.wav2vec2 = Wav2Vec2Model(config)
#         self.args = {}
        
#         self.init_weights()
    
#     def freeze_feature_extractor(self):
#         self.wav2vec2.feature_extractor._freeze_parameters()
    
#     def forward(self, x, attention_mask=None):
#         outputs = self.wav2vec2(x, attention_mask=attention_mask)
#         return outputs['extract_features']

class WAV2VEC2_MODIFIED_SHORTFORM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # config = AutoConfig.from_pretrained(
        #     "kresnik/wav2vec2-large-xlsr-korean",
        #     num_labels = args.num_labels,
        #     finetuning_task = "wav2vec2_clf"
        # )
        # setattr(config, 'pooling_mode', args.pooling_mode)

        # self.num_labels = args.num_labels

        # self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean", config=config)

        self.ff1 = nn.Linear(79872, 512)
        self.ff2 = nn.Linear(512, self.num_labels)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.1)
        
        # self.audio_feature_extractor.init_weights()

    def forward(self, x, attention_mask=None):
        output = self.dp1(self.bn1(F.relu(self.ff1(x.view(16, -1)))))
        output = self.ff2(output)
        return output