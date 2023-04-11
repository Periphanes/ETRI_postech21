import os
import random

import torch

import pickle
from tqdm import tqdm

from transformers import Wav2Vec2Processor
import torchaudio

class binary_static_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)
                
                # Discard Results with more than one answer
                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["text"] == None:
                    continue

                self._data_list.append(pkl_path)
        
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        return self._data_list[index]

class wav2vec2_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type='dataset'):
        self._data_list = []

        if os.path.exists(os.path.join(os.getcwd(), 'wav2vec_processor.pickle')):
            with open('wav2vec_processor.pickle', 'rb') as file:
                self.processor = pickle.load(file)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
            with open('wav2vec_processor.pickle', 'wb') as file:
                pickle.dump(self.processor, file, pickle.HIGHEST_PROTOCOL)
        
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                speech_array, sampling_rate = torchaudio.load(data_point['wav_dir'])
                resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()

                speech_feature = self.processor.__call__(audio=speech, sampling_rate=self.target_sampling_rate)

                self._data_list.append((data_point['total_emot'][0], speech_feature))
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]