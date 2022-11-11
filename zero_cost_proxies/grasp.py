import torch
from utils_functions import initialise_zero_cost_proxy, get_score
import torch.autograd as autograd

def calculate_grasp(net, data_loader, hyperparameters, output_device, loss_function ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=False, eval=True)
    
    weights = []
    for layer in net.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)

    N = data.shape[0]
    
    #forward/grad pass #1
    
    outputs = net.forward(data)
    loss = loss_function(outputs, labels)
    grad_w_p = autograd.grad(loss, weights, allow_unused=True)
    grad_w = list(grad_w_p)
    

    # forward/grad pass #2
    outputs = net.forward(data)
    loss = loss_function(outputs, labels)
    grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
    
    # accumulate gradients computed in previous step and call backwards
    z, count = 0,0
    for layer in net.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            if grad_w[count] is not None:
                z += (grad_w[count].data * grad_f[count]).sum()
            count += 1
    z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    
    score = get_score(model, grasp, "param")
    return score

