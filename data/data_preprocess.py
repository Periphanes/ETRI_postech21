import torch
import random
import numpy as np

import os
import pickle
from tqdm import tqdm

from data.collate_fn import *
from data.dataset import *
from torch.utils.data import DataLoader

def get_data_loader(args):

    print("Initializing Data Loader and Datasets")

    session_ids = [i for i in range(1,21)]
    random.shuffle(session_ids)

    train_ids = session_ids[:16]
    test_ids = session_ids[16:]

    train_data_list = []
    test_data_list = []

    file_dir = os.listdir(os.path.join(os.getcwd(), 'dataset/processed'))
    for data_file in file_dir:
        data_session_id = int(data_file.split("/")[-1][:2])
        if data_session_id in train_ids:
            train_data_list.append(data_file)
        else:
            test_data_list.append(data_file)
    
    random.shuffle(train_data_list)
    val_len = int(float(len(train_data_list)) / 4)
    val_data_list = train_data_list[:val_len]
    train_data_list = train_data_list[val_len:]

    if args.trainer == "binary_classification_static":
        train_data      = binary_static_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = binary_static_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = binary_static_Dataset(args, data=test_data_list, data_type="testing dataset")

    train_loader = DataLoader(  train_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_static)
    val_loader = DataLoader(    val_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_static)
    test_loader = DataLoader(   test_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_static)

    return train_loader, val_loader, test_loader