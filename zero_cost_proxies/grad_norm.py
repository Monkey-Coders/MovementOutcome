import torch
from utils_functions import sum_arr, get_layer_metric_array
from torch.autograd import Variable



def calculate_grad_norm(model, data_loader, hyperparameters, output_device, loss_function ):
    new_model = model.get_copy().to("cuda")
    new_model.train()

    loader = data_loader['train']
    process = iter(loader)
    batch = next(process)
    data, labels, video_ids, indices = batch
    if hyperparameters['devices']['gpu_available']:
        data = Variable(data.float().cuda(output_device), requires_grad=False) 
        labels = Variable(labels.long().cuda(output_device), requires_grad=False)
    else:
        data = Variable(data.float(), requires_grad=False) 
        labels = Variable(labels.long(), requires_grad=False)

    
    output, _ = new_model(data)
    loss = loss_function(output, labels)
    loss.backward()
    grad_norm_arr = get_layer_metric_array(new_model, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
    score = sum_arr(grad_norm_arr)
    return score