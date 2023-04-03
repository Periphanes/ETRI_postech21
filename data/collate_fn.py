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

        y = data_point["total_emot"][0]
        y_batch.append(y)
    
    X = torch.stack(X_batch)
    y = torch.tensor(y_batch)

    return X, y

def collate_txt(train_data):
    X_batch_id = []
    X_batch_attention = []
    y_batch = []

    for pkl_id, pkl_path in enumerate(train_data):
        with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
            data_point = pickle.load(f)

        X_id = data_point["input_ids"]
        X_attention = data_point["attention_mask"]

        X_batch_id.append(X_id)
        X_batch_attention.append(X_attention)

        y = data_point["total_emot"][0]
        y_batch.append(y)
    
    X_ids = torch.stack(X_batch_id)
    X_attention_mask = torch.stack(X_batch_attention)
    y = torch.tensor(y_batch)

    return (X_ids, X_attention_mask), y