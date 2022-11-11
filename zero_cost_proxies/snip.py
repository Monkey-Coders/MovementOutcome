import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_functions import sum_arr, get_layer_metric_array

import types

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def calculate_snip(model, data_loader, hyperparameters, output_device, loss_function ):
   