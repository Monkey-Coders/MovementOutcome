import torch
import numpy as np
from utils_functions import initialise_zero_cost_proxy, get_score


def get_batch_jacobian(net, x, target):
    net.zero_grad()
    x.requires_grad = True
    out, features = net(x)
    out.backward(torch.ones_like(out))
    jacob = x.grad.detach()
    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


def calculate_jacobian(net, data_loader, hyperparameters, output_device, loss_function ):
    bn = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data, labels, loader = initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, train=False, eval=True, bn=bn)

    try:
        jacobs, labels = get_batch_jacobian(model, data, labels)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan
    return jc