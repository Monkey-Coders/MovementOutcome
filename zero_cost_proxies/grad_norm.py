import torch
from utils_functions import initialise_zero_cost_proxy, get_score
from torch.autograd import Variable



def calculate_grad_norm(net, data_loader, hyperparameters, output_device, loss_function ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=True, eval=False)

    output, _ = model(data)
    loss = loss_function(output, labels)
    loss.backward()
    score = get_score(model, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
    return score