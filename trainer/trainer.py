import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from control.config import args

def binary_classification_static(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x = x.type(torch.FloatTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x)
        output = output.squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

    elif flow_type == "val":
        output = model(x)
        output = output.squeeze()

        loss = criterion(output, y)

    else:
        output = model(x)
        output = output.squeeze()

        return output, y
    
    return model, loss.item()

def classification_with_txt_static(args, iteration, x1, x2, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x1 = x1.type(torch.LongTensor).to(device)
    x2 = x2.type(torch.LongTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x1, x2)
        output = output.squeeze()
        
        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

    elif flow_type == "val":
        output = model(x1, x2)
        output = output.squeeze()

        loss = criterion(output, y)

    else:
        output = model(x1, x2)
        output = output.squeeze()

        return output, y
    
    return model, loss.item()

def classification_audio(args, iteration, x, attention, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x = x.type(torch.FloatTensor).to(device)
    attention = attention.type(torch.LongTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x, attention_mask=attention)

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), 5)

        optimizer.step()
        scheduler.step()
    
    elif flow_type == "val":
        output = model(x, attention_mask=attention)

        loss = criterion(output, y)

    else:
        output = model(x, attention_mask=attention)

        return output, y

    return model, loss.item()

def classification_audio_txt(args, iteration, x_audio, x_audio_attn, x_txt, x_txt_attn, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x_audio = x_audio.type(torch.FloatTensor).to(device)
    x_audio_attn = x_audio_attn.type(torch.LongTensor).to(device)
    x_txt = x_txt.type(torch.LongTensor).to(device)
    x_txt_attn = x_txt_attn.type(torch.LongTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x_audio, x_audio_attn, x_txt, x_txt_attn)
        output = output.squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

    elif flow_type == "val":
        output = model(x_audio, x_audio_attn, x_txt, x_txt_attn)
        output = output.squeeze()

        loss = criterion(output, y)

    else:
        output = model(x_audio, x_audio_attn, x_txt, x_txt_attn)
        output = output.squeeze()

        return output, y
    
    return model, loss.item()