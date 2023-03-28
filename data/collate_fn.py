import pickle
import os

import torch
import random
import numpy as np
import torch.nn.functional as F

from control.config import args

def collate_static(train_data):
    X_batch = []
    y_batch = []

    for pkl_id, pkl_path in enumerate(train_data):
        with open(os.path.join('dataset/processed', 'rb')) as f:
            data_point = pickle.load(f)
        
        print(data_point)
        exit(0)