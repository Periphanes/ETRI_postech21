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

    if args.datasets == "default":
        session_ids = [i for i in range(1,21)]
        random.shuffle(session_ids)

        train_ids = session_ids[:16]
        test_ids = session_ids[16:]
    
    if args.datasets == "augmented":
        session_ids = [i for i in range(1,21)]
        random.shuffle(session_ids)

        session_ids_20 = [i for i in range(21,41)]
        random.shuffle(session_ids_20)

        train_ids = session_ids[:16] + session_ids_20[:16]
        test_ids = session_ids[16:] + session_ids_20[16:]

    if args.small_dataset == True:
        train_ids = [7,8]
        test_ids = [5]

    train_data_list = []
    test_data_list = []

    file_dir = os.listdir(os.path.join(os.getcwd(), 'dataset/processed'))
    for data_file in file_dir:
        data_session_id = int(data_file.split("/")[-1][4:6])
        if data_session_id in train_ids:
            train_data_list.append(data_file)
        elif data_session_id in test_ids:
            test_data_list.append(data_file)
        else:
            div = ord(data_file.split("/")[-1][-8]) % 5
            if div:
                train_data_list.append(data_file)
            else:
                test_data_list.append(data_file)
    
    random.shuffle(train_data_list)
    val_len = int(float(len(train_data_list)) / 4)
    val_data_list = train_data_list[:val_len]
    train_data_list = train_data_list[val_len:]

    if args.trainer == "binary_classification_static" or args.trainer == "classification_with_txt_static":
        train_data      = binary_static_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = binary_static_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = binary_static_Dataset(args, data=test_data_list, data_type="testing dataset")
    if args.trainer == "classification_audio":
        train_data      = wav2vec2_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = wav2vec2_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = wav2vec2_Dataset(args, data=test_data_list, data_type="testing dataset")
    if args.trainer == "classification_audio_txt":
        train_data      = audio_txt_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = audio_txt_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = audio_txt_Dataset(args, data=test_data_list, data_type="testing dataset")
    if args.trainer == "classification_audio_txt_shortform":
        train_data      = audio_txt_shortform_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = audio_txt_shortform_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = audio_txt_shortform_Dataset(args, data=test_data_list, data_type="testing dataset")
    if args.trainer == "classification_txt_shortform":
        train_data      = txt_shortform_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = txt_shortform_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = txt_shortform_Dataset(args, data=test_data_list, data_type="testing dataset")
        

    print("Total of {} data points intialized in Training Dataset...".format(train_data.__len__()))
    print("Total of {} data points intialized in Validation Dataset...".format(val_data.__len__()))
    print("Total of {} data points intialized in Testing Dataset...".format(test_data.__len__()))

    if args.input_types == "txt":
        train_loader = DataLoader(  train_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_txt)
        val_loader = DataLoader(    val_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_txt)
        test_loader = DataLoader(   test_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_txt)
    elif args.input_types == "audio":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                                collate_fn=collate_audio)
        test_loader = DataLoader(test_data, batch_size=args.batch_size,drop_last=True,
                                 collate_fn=collate_audio)
    elif args.input_types == "audio_txt":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt)
    elif args.input_types == "audio_txt_shortform":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt_shortform)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt_shortform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_audio_txt_shortform)
    elif args.input_types == "txt_shortform":
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_txt_shortform)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_txt_shortform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                                  collate_fn=collate_txt_shortform)
    else:
        train_loader = DataLoader(  train_data, batch_size=args.batch_size, drop_last=True,
                                    collate_fn=collate_static)
        val_loader = DataLoader(    val_data, batch_size=args.batch_size, drop_last=True,
                                    collate_fn=collate_static)
        test_loader = DataLoader(   test_data, batch_size=args.batch_size, drop_last=True,
                                    collate_fn=collate_static)

    return train_loader, val_loader, test_loader