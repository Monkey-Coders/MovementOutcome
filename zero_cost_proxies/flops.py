import torch
from torchprofile import profile_macs
import numpy as np

from utils_functions import initialise_zero_cost_proxy 

def calculate_flops(net, data_loader, hyperparameters, output_device, loss_function ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=True, eval=False)
    input_channels = 6
    dummy_data = torch.from_numpy(np.zeros((2, input_channels, hyperparameters['model']['input_temporal_resolution'], hyperparameters['model']['input_spatial_resolution']))).float()
    if hyperparameters['devices']['gpu_available']:
        dummy_data = dummy_data.cuda(output_device)
    

    macs = profile_macs(model, dummy_data) // 2
    num_flops = int(macs*2)
    return num_flops
