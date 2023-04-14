import os
import random

import torch

import pickle
from tqdm import tqdm

class binary_static_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []
        
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)
                
                # Discard Results with more than one answer
                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["text"] == None or "input_ids" not in data_point:
                    continue

                self._data_list.append(pkl_path)
        
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        return self._data_list[index]

class wav2vec2_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type='dataset'):
        self._data_list = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                self._data_list.append((data_point['total_emot'][0], data_point['wav_vector']))
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]

class audio_shortform_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type='dataset'):
        self._data_list = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                self._data_list.append((data_point['audio_output'], data_point['total_emot'][0]))
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]


class audio_txt_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                if data_point["text"] == None:
                    continue

                data_sample = (data_point['wav_vector'], data_point["input_ids"], data_point["attention_mask"], data_point["total_emot"][0])

                # self._data_list.append((data_point['total_emot'][0], speech_feature))
                self._data_list.append(data_sample)
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]

class audio_txt_shortform_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                if data_point["text"] == None:
                    continue

                if "text_output" not in data_point:
                    continue

                data_sample = (data_point['audio_output'], data_point["text_output"], data_point["total_emot"][0])

                # self._data_list.append((data_point['total_emot'][0], speech_feature))
                self._data_list.append(data_sample)
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]

class txt_shortform_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []

        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if len(data_point["total_emot"]) > 1:
                    continue

                if data_point["wav_dir"][-3:] != "wav":
                    continue

                if data_point["text"] == None:
                    continue

                if "text_output" not in data_point:
                    continue

                data_sample = (data_point["text_output"], data_point["total_emot"][0])

                # self._data_list.append((data_point['total_emot'][0], speech_feature))
                self._data_list.append(data_sample)
    
    def __len__(self):
        return len(self._data_list)

    def __getitem__(self,index):
        return self._data_list[index]