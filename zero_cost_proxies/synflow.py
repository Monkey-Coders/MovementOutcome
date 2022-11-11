import torch
from utils_functions import initialise_zero_cost_proxy, get_score

# torch.set_default_dtype(torch.float64)

def calculate_synflow(net, data_loader, hyperparameters, output_device, loss_function ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=False, eval=True)

    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs


    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    signs = linearize(model)

    model.zero_grad()
    model.double()
    input_dim = list(data[0,:].shape)

    inputs = torch.ones([1] + input_dim).double().to(device)
    outputs,_ = model.forward(inputs)
    torch.sum(outputs).backward()

    def synflow(layer):
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)

    score = get_score(model, synflow, "param")
    return score

