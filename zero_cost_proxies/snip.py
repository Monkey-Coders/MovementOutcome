import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_functions import sum_arr, get_layer_metric_array
from torch.autograd import Variable

import types

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def calculate_snip(net, data_loader, hyperparameters, output_device, loss_function ):
    model = net.get_copy().to("cuda")
    model.train()

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    model.zero_grad()
    loader = data_loader['train']
    process = iter(loader)
    batch = next(process)
    data, labels, video_ids, indices = batch
    N = data.shape[0]
    if hyperparameters['devices']['gpu_available']:
        data = Variable(data.float().cuda(output_device), requires_grad=False) 
        labels = Variable(labels.long().cuda(output_device), requires_grad=False)
    else:
        data = Variable(data.float(), requires_grad=False) 
        labels = Variable(labels.long(), requires_grad=False)
    
    outputs, _ = model.forward(data)
    loss = loss_function(outputs, labels)
    loss.backward()

    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)
    
    grad_abs = get_layer_metric_array(model, snip, "param")
    sum = sum_arr(grad_abs)

    return sum
   