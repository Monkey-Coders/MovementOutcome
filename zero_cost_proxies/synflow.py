import torch
from utils_functions import sum_arr, get_layer_metric_array

# torch.set_default_dtype(torch.float64)

def calculate_synflow(model, data_loader, hyperparameters, output_device, loss_function ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    new_model = model.get_copy().to(device)
    new_model.eval()
    #new_model.train()
    loader = data_loader['train']

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

    # keep signs of all params
    signs = linearize(new_model)

    model.zero_grad()
    model = model.float()

    for param in model.parameters():
        print(param.dtype)

    process = iter(loader)
    batch = next(process)
    data, labels, video_ids, indices = batch

    input_dim = list(data[0,:].shape)
    data = torch.ones([1] + input_dim).float().to(device)
    print(data.dtype)
    print(data.shape)

    # data = data.double()
    output, _ = new_model.forward(data)
    torch.sum(output).backward()


    

    def synflow(layer):
        if layer.weight.grad is not None:
            print("Weight grad")
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)
    
    grad_abs = get_layer_metric_array(model, synflow, mode='param' )
    # apply signs of all params
    nonlinearize(model, signs)
    score = sum_arr(grad_abs)

    return score

