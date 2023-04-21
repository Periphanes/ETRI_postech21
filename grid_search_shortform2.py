# Main Training File
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from tqdm import tqdm

from control.config import args
from models import get_model
from trainer import get_trainer
from data.data_preprocess import get_data_loader

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from transformers import AutoConfig

log_directory = os.path.join(args.dir_result, args.project_name)

# make sure that CUDA uses GPU according to the inserted order
# not useful unless in multi-GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

args.seed = args.seed_list[0]
# set all the seeds in random and other libraries to given seed_num
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Setting flags to make sure reproducability for certain seeds
# cudnn.deterministic weeds out random algorithms,
# cudnn.benchmark disables benchmarking, which may introduce randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set main device to CPU or GPU(cuda)
if args.cpu or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

print("Device Used : ", device)
args.device = device

if args.input_types == "audio_txt_shortform":
    args.trainer = "classification_audio_txt_shortform"
elif args.input_types == "txt_shortform":
    args.trainer = "classification_txt_shortform"
elif args.input_types == "audio_shortform":
    args.trainer = "classification_audio_shortform"
else:
    raise NotImplementedError("Trainer Not Implemented Yet")

train_loader, val_loader, test_loader = get_data_loader(args)

hyper_f1_score_lst = [0.0]
best_parameters = []

# args.audio_weight = 1.0
# hyper_audio_weight = args.audio_weight
args.txt_weight = 1.0
hyper_txt_weight = args.txt_weight
args.bottleneck_length = 4
bottleneck_length = args.bottleneck_length
args.transformer_layers = 4
transformer_layers = args.transformer_layers
args.transformer_heads = 4
transformer_heads = args.transformer_heads
args.transformer_ff_dim = 1024
transformer_ff_dim = args.transformer_ff_dim
args.transformer_dropout = 0.12
transformer_dropout = args.transformer_dropout
args.transformer_activation = 'gelu'
transformer_activation = args.transformer_activation
candidate_lst = [i for i in range(100)]
random.shuffle(candidate_lst)
data = {'audio_weight': [], 'f1_score': [], 'precision_score': [], 'recall_score': []}
for candidate_value in candidate_lst:
    hyper_audio_weight = 0.01 * (1.096478 ** candidate_value)
    args.audio_weight = hyper_audio_weight
    data['audio_weight'].append(candidate_value)

    model = get_model(args)
    model = model(args).to(device)

    frequency = np.array([1, 1, 1, 1, 1, 1, 1])
    frequency = 1 / frequency
    frequency = frequency / sum(frequency) * 7
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=torch.FloatTensor(frequency).to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_init)

    iter_num_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    iteration = 0

    best_validation_loss = 100

    for epoch in tqdm(range(1, args.epochs+1)):

        # Training Step Start
        model.train()

        training_loss = []

        for train_batch in train_loader:
            if args.trainer == "classification_audio_txt_shortform":
                train_x, train_y = train_batch
                train_x = (train_x[0].to(device), train_x[1].to(device))
                train_y = train_y.to(device)
            elif args.trainer == "classification_txt_shortform":
                train_x, train_y = train_batch
                train_x = train_x.to(device)
                train_y = train_y.to(device)
            elif args.trainer == "classification_audio_shortform":
                train_x, train_y = train_batch
                train_x = train_x.to(device)
                train_y = train_y.to(device)

            iteration += 1

            model, iter_loss = get_trainer(args = args,
                                        iteration = iteration,
                                        x = train_x,
                                        static = None,
                                        y = train_y,
                                        model = model,
                                        device = device,
                                        scheduler=scheduler,
                                        optimizer=optimizer,
                                        criterion=criterion,
                                        flow_type="train")

            training_loss.append(iter_loss)

        # Validation Step Start
        model.eval()

        validation_loss = []

        with torch.no_grad():
            pred_batches = []
            true_batches = []
            for val_batch in val_loader:
                if args.trainer == "classification_audio_txt_shortform":
                    val_x, val_y = val_batch
                    val_x = (val_x[0].to(device), val_x[1].to(device))
                    val_y = val_y.to(device)
                elif args.trainer == "classification_txt_shortform":
                    val_x, val_y = val_batch
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                elif args.trainer == "classification_audio_shortform":
                    val_x, val_y = val_batch
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)

                pred, val_loss = get_trainer(args = args,
                                                iteration = iteration,
                                                x = val_x,
                                                static = None,
                                                y = val_y,
                                                model = model,
                                                device = device,
                                                scheduler=scheduler,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                flow_type="val")
                pred_batches.append(pred)
                true_batches.append(val_y)

                validation_loss.append(val_loss)

        pred = torch.argmax(torch.cat(pred_batches), dim=1).cpu()
        true = torch.cat(true_batches).cpu()
        now_validation_loss = sum(validation_loss) / len(validation_loss)
        if best_validation_loss > now_validation_loss:
            torch.save(model, './saved_models/best_model.pt')
            best_validation_loss = now_validation_loss

    # Test Step Start
    model = torch.load('./saved_models/best_model.pt').to(device)
    model.eval()
    with torch.no_grad():
        pred_batches = []
        true_batches = []

        for test_batch in test_loader:
            if args.trainer == "classification_audio_txt_shortform":
                test_x, test_y = test_batch
                test_x = (test_x[0].to(device), test_x[1].to(device))
                test_y = test_y.to(device)
            elif args.trainer == "classification_txt_shortform":
                test_x, test_y = test_batch
                test_x = test_x.to(device)
                test_y = test_y.to(device)
            elif args.trainer == "classification_audio_shortform":
                test_x, test_y = test_batch
                test_x = test_x.to(device)
                test_y = test_y.to(device)

            pred, true = get_trainer(args = args,
                                    iteration = iteration,
                                    x = test_x,
                                    static = None,
                                    y = test_y,
                                    model = model,
                                    device = device,
                                    scheduler=scheduler,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    flow_type="test")

            pred_batches.append(pred)
            true_batches.append(test_y)

    pred = torch.argmax(torch.cat(pred_batches), dim=1).cpu()
    true = torch.cat(true_batches).cpu()
    temp_f1_score = f1_score(true, pred, average='weighted')
    temp_precision_score = precision_score(true, pred, average='weighted')
    temp_recall_score = recall_score(true, pred, average='weighted')
    print(temp_f1_score)
    print(temp_precision_score)
    print(temp_recall_score)
    data['f1_score'].append(temp_f1_score)
    data['precision_score'].append(temp_precision_score)
    data['recall_score'].append(temp_recall_score)

    if temp_f1_score > max(hyper_f1_score_lst):
        best_parameters = [hyper_audio_weight, bottleneck_length, transformer_layers, transformer_heads, transformer_ff_dim, transformer_dropout]
        print(best_parameters)
        print("*** The best : ", end="")
        print(temp_f1_score)

    hyper_f1_score_lst.append(temp_f1_score)

print(best_parameters)

df = pd.DataFrame(data)
df.to_csv("data.csv")