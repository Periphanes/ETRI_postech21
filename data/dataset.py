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

                self._data_list.append(pkl_path)
        
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        return self._data_list[index]

