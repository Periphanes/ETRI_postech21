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
        with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
            data_point = pickle.load(f)

        X = [0,0,0]
        X[0] = 0 if data_point["session_gen"] == "M" else 1
        X[1] = len(data_point["text"])
        X[2] = len(data_point["text"].split())

        X_batch.append(torch.tensor(X))

        y = F.one_hot(torch.tensor(data_point["total_emot"][0]), 7)
        y = y.squeeze()
        y_batch.append(y)
    
    X = torch.stack(X_batch)
    y = torch.stack(y_batch)

    return X, y