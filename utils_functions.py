import json
model_script = 'models.gcn_search_model'

m = __import__(model_script, fromlist=['object'])
import os 
from utils import graph
import torch
import torch.nn as nn
from torch.autograd import Variable



body_parts= ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe', 'left_big_toe']
# body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'] # <-- Assign body parts in skeleton (e.g., In-Motion 19: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'], In-Motion 29: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe', 'left_big_toe'])
neighbor_link = [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5,8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14), (19,7), (20,7), (21,11), (22,11), (23,15), (24,15), (25,15), (26,18), (27,18), (28,18)]
center = 8 # <-- Assign index of body part in center of skeleton (e.g., In-Motion 19/29: 8 for thorax) # <-- Assign neighboring body parts that are connected by bones in the skeleton (e.g., In-Motion 19: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5, 8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14)], In-Motion 29: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5,8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14), (19,7), (20,7), (21,11), (22,11), (23,15), (24,15), (25,15), (26,18), (27,18), (28,18)])
#In-Motion 29:
bone_conns = [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18, 18]
# bone_conns = [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17] # <-- Assign parent (i.e., body part closer to the center) of each body part (e.g., In-Motion 19: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17], In-Motion 29: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18, 18])
thorax_index = 8 # <-- Assign index of thorax body part (e.g., In-Motion 19/29: 8)
pelvis_index = 12 # <-- Assign index of pelvis body part (e.g., In-Motion 19/29: 12)
use_mask = True 
sample_coords = [(0.51, 0.07), (0.51, 0.15), (0.473, 0.187), (0.55, 0.183), (0.51, 0.22), (0.47, 0.26), (0.425, 0.28), (0.383, 0.254), (0.515, 0.262), (0.565, 0.265), (0.605, 0.265), (0.637, 0.252),(0.508, 0.43), (0.47, 0.42), (0.453, 0.562), (0.465, 0.70), (0.546, 0.44), (0.532, 0.585), (0.50, 0.72)] # <-- Assign dummy coordinates for each body part


def get_candidate(file_path):
    with open(os.path.join(file_path, 'candidate.json'), 'r') as json_file: 
        try:
            candidate = json.load(json_file)
        except json.decoder.JSONDecodeError:
            return None
    return candidate

def get_model_from_candidate_file(file_path):
    with open(os.path.join(file_path, 'candidate.json'), 'r') as json_file: 
        candidate = json.load(json_file)


    if candidate['graph'] == 'spatial':
        labeling_mode = ['spatial']
        disentangled_num_scales = [None]
    elif candidate['graph'].startswith('dis'):
        labeling_mode = ['disentangled']
        if candidate['graph'] == 'dis2':
            disentangled_num_scales = [2]
        elif candidate['graph'] == 'dis4':
            disentangled_num_scales = [4]
        elif candidate['graph'] == 'dis4+2':
            labeling_mode += ['disentangled']
            disentangled_num_scales = [4, 2]
    graph_input = graph.Graph(strategy=labeling_mode[0], body_parts=body_parts, neighbor_link=neighbor_link, center=center, bone_conns=bone_conns, thorax_index=thorax_index, pelvis_index=pelvis_index, disentangled_num_scales=disentangled_num_scales[0], use_mask=use_mask)
    if len(labeling_mode) == 2:
        graph_main = graph.Graph(strategy=labeling_mode[1], body_parts=body_parts, neighbor_link=neighbor_link, center=center, bone_conns=bone_conns, thorax_index=thorax_index, pelvis_index=pelvis_index, disentangled_num_scales=disentangled_num_scales[1], use_mask=use_mask)


    print(candidate)

    print("----------")
    # Determine model architecture
    input_width = candidate['input_width']
    num_input_modules = candidate['num_input_modules']
    initial_block_type = candidate['initial_block_type']
    initial_residual = candidate['initial_residual']
    num_input_modules = candidate['num_input_modules']
    num_input_modules = candidate['num_input_modules']
    num_input_modules = candidate['num_input_modules']
    if candidate['input_temporal_scales'] == 'linear':
        input_temporal_scales = [i for i in range(1,num_input_modules+1)]
    else:
        input_temporal_scales = [int(candidate['input_temporal_scales']) for i in range(num_input_modules)]
    initial_main_width = candidate['initial_main_width']
    num_main_levels = candidate['num_main_levels']
    num_main_level_modules = candidate['num_main_level_modules']
    block_type = candidate['block_type']
    bottleneck_factor = candidate['bottleneck_factor']
    residual = candidate['residual']
    main_temporal_scales = candidate['bottleneck_factor']
    if candidate['main_temporal_scales'] == 'linear':
        main_temporal_scales = [i for i in range(1,num_main_levels+1)]
    else:
        main_temporal_scales = [int(candidate['main_temporal_scales']) for i in range(num_main_levels)]
    temporal_kernel_size = candidate['temporal_kernel_size']
    se_outer = False
    se_inner = False
    if candidate['se'] in ['inner', 'both']:
        se_inner = True
    if candidate['se'] in ['outer', 'both']:
        se_outer = True
    se_ratio = candidate['se_ratio']
    relative_se = True if candidate['se_type'] == 'relative' else False
    swish_nonlinearity = True if candidate['nonlinearity'] == 'swish' else False
    attention = candidate['attention']
    spatial_pool = True if candidate['pool'] == 'spatial' else False
            
    # Set main processing unit
    output_device = 0
    output_device = int(output_device)
    gpu_available = torch.cuda.is_available()

    edge_importance_weighting = True
    dropout = 0
    num_input_branches = 3

    #TODO:
    # Find out if this is correct or not
    input_channels = 6
    # Initialize model
    model = m.Model(num_classes=2, graphs=[graph_input, graph_main] if len(labeling_mode) == 2 else [graph_input], input_channels=input_channels, edge_importance_weighting=edge_importance_weighting, dropout=dropout, num_input_branches=num_input_branches, attention=attention, se_outer=se_outer, se_inner=se_inner, initial_residual=initial_residual, residual=residual, initial_block_type=initial_block_type, block_type=block_type, input_width=input_width, initial_main_width=initial_main_width, temporal_kernel_size=temporal_kernel_size, num_input_modules=num_input_modules, num_main_levels=num_main_levels, num_main_level_modules=num_main_level_modules, input_temporal_scales=input_temporal_scales, main_temporal_scales=main_temporal_scales, bottleneck_factor=bottleneck_factor, se_ratio=se_ratio, relative_se=relative_se, swish_nonlinearity=swish_nonlinearity, spatial_pool=spatial_pool)
    if gpu_available:
        model = model.cuda(output_device)

    return model


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))
    
    return metric_array


def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def get_score(net, metric, mode, is_jacob_cov = False):
    metric_array = get_layer_metric_array(net, metric, mode)
    if is_jacob_cov:
        return metric_array
    return sum_arr(metric_array)

def initialise_zero_cost_proxy(net, data_loader, hyperparameters, output_device, eval = False, train = True,  single_batch = True, bn = False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = net.get_copy(bn=bn).to(device)
    model.zero_grad()
    if train:
        model.train()
    if eval:
        model.eval()
    
    loader = data_loader["train"]

    if single_batch:
        process = iter(loader)
        batch = next(process)
        data, labels, video_ids, indices = batch
    
    if hyperparameters['devices']['gpu_available']:
        data = Variable(data.float().cuda(output_device), requires_grad=False) 
        labels = Variable(labels.long().cuda(output_device), requires_grad=False)
    else:
        data = Variable(data.float(), requires_grad=False) 
        labels = Variable(labels.long(), requires_grad=False)

    return model, data, labels, loader


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)