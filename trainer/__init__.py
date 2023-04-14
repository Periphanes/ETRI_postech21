from .trainer import *


def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.trainer == "binary_classification_static":
        model, iter_loss = binary_classification_static(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_with_txt_static":
        model, iter_loss = classification_txt(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_audio":
        model, iter_loss = classification_audio(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_audio_txt":
        model, iter_loss = classification_audio_txt(args, iteration, x[0], x[1], x[2], x[3], y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_audio_txt_shortform":
        model, iter_loss = classification_audio_txt_shortform(args, iteration, x[0], x[1], y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_txt_shortform":
        model, iter_loss = classification_txt_shortform(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    elif args.trainer == "classification_audio_shortform":
        model, iter_loss = classification_audio_shortform(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected Trainer is Not Prepared Yet...")
        exit(1)

    return model, iter_loss
