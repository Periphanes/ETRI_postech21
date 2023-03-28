from .trainer import *

def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.predict_type == "static":
        model, iter_loss = binary_classification_static(args, iteration, x, static, y,model, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected Trainer is Not Prepared Yet...")
        exit(1)
    
    return model, iter_loss