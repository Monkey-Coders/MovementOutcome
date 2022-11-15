import torch
import torch.nn as nn

from utils_functions import initialise_zero_cost_proxy

def calculate_params(net, data_loader, hyperparameters, output_device, loss_function ):
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=True, eval=False)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return num_parameters
