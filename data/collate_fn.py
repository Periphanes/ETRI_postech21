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
        X[0] = 0
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

def collate_audio(train_data):
    X_batch = []
    X_batch_attention = []
    y_batch = []

    for data_point in train_data:
        audio_sample = data_point[1]['input_values'][0]
        audio_attention_mask = data_point[1]['attention_mask'][0]
        audio_label = data_point[0]

        if len(audio_sample) > args.audio_max_length:
            audio_sample = audio_sample[:args.audio_max_length]
            audio_attention = np.zeros(args.audio_max_length)
        else:
            audio_attention = np.concatenate((np.zeros(len(audio_sample)), np.ones(args.audio_max_length - len(audio_sample))))
            audio_sample = np.concatenate((audio_sample, np.zeros(args.audio_max_length - len(audio_sample))))
        
        X = torch.Tensor(audio_sample)
        X_attention = torch.Tensor(audio_attention)

        X_batch.append(X)
        X_batch_attention.append(X_attention)
        y_batch.append(audio_label)

    X = torch.stack(X_batch)
    X_attention = torch.stack(X_batch_attention)
    y = torch.Tensor(y_batch)

    return (X, X_attention), y

def collate_audio_txt(train_data):
    X_audio_batch = []
    X_audio_attention_batch = []
    X_txt_batch = []
    X_txt_attention_batch = []
    y_batch = []

    for data_point in train_data:
        audio_sample = data_point[0]['input_values'][0]
        label = data_point[0]

        if len(audio_sample) > args.audio_max_length:
            audio_sample = audio_sample[:args.audio_max_length]
            audio_attention = np.zeros(args.audio_max_length)
        else:
            audio_attention = np.concatenate((np.zeros(len(audio_sample)), np.ones(args.audio_max_length - len(audio_sample))))
            audio_sample = np.concatenate((audio_sample, np.zeros(args.audio_max_length - len(audio_sample))))
        
        X_audio = torch.Tensor(audio_sample)
        X_audio_attention = torch.Tensor(audio_attention)

        X_txt = data_point[1]
        X_txt_attention = data_point[2]

        X_audio_batch.append(X_audio)
        X_audio_attention_batch.append(X_audio_attention)
        X_txt_batch.append(X_txt)
        X_txt_attention_batch.append(X_txt_attention)

        y = data_point[3]
        y_batch.append(y)
    
    X_audio = torch.stack(X_audio_batch)
    X_audio_attention = torch.stack(X_audio_attention_batch)
    X_txt = torch.stack(X_txt_batch)
    X_txt_attention = torch.stack(X_txt_attention_batch)
    y = torch.tensor(y_batch)

    return (X_audio, X_audio_attention, X_txt, X_txt_attention), y

def collate_audio_txt_shortform(train_data):
    X_audio_batch = []
    X_txt_batch = []
    y_batch = []

    for data_point in train_data:
        X_audio_batch.append(data_point[0].squeeze())
        X_txt_batch.append(data_point[1].squeeze())
        y_batch.append(data_point[2])
    
    X_audio = torch.stack(X_audio_batch)
    X_txt = torch.stack(X_txt_batch)
    y = torch.tensor(y_batch)

    return (X_audio, X_txt), y

def collate_txt_shortform(train_data):
    X_batch = []
    y_batch = []

    for data_point in train_data:
        X_batch.append(data_point[0].squeeze())
        y_batch.append(data_point[1])
    
    X_txt = torch.stack(X_batch)
    y = torch.tensor(y_batch)

    return X_txt, y

def collate_audio_shortform(train_data):
    X_audio_batch = []
    y_batch = []

    for data_point in train_data:
        X_audio_batch.append(data_point[0].squeeze())
        y_batch.append(data_point[1])
    
    X_audio = torch.stack(X_audio_batch)
    y = torch.tensor(y_batch)

    return X_audio, y