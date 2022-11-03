import os, sys
import json
import csv
import math
import torch
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from utils import graph
search_name = '21092022 1522 IM2021'
project_name = 'test' # <---Enter the name of your project folder

model_script = 'models.gcn_search_model'

m = __import__(model_script, fromlist=['object'])

crossval_folds = 7

project_dir = os.path.join('projects', project_name)
sys.path.append(project_dir)
sys.path.append(os.path.join('.'))

#### Step 6: Choose graph type and configuration.
# In-Motion_ 29:
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



""" Define directories """
  
# search_dir = os.path.join(project_dir, 'searches', search_name)
experiments_dir = os.path.join(project_dir, 'experiments')
""" results_dir = os.path.join(search_dir, coords_dir.split('/')[-1])
os.makedirs(results_dir, exist_ok=True) """

candidate_performance = 0.0
candidate_num = None
for experiment_dir in os.listdir(experiments_dir):
    if experiment_dir.startswith('search_'):
        with open(os.path.join(experiments_dir, experiment_dir, 'validation_results.json'), 'r') as json_file:  
            validation_results = json.load(json_file)
            if validation_results['best_auc'] > candidate_performance:
                candidate_performance = validation_results['best_auc']
                candidate_num = experiment_dir.split('_')[1]
with open(os.path.join(experiments_dir, 'search_{0}'.format(candidate_num), 'candidate.json'), 'r') as json_file: 
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


# Fetch epoch with highest performance

experiment_dir = f"projects/test/experiments/search_r2/"
with open(os.path.join(experiment_dir, 'validation_results.json'), 'r') as json_file: 
    #best_epoch = json.load(json_file)['best_auc_epoch']
    best_epoch = json.load(json_file)['best_loss_epoch']
            
# Load model weights
for file in os.listdir(experiment_dir):
    if file.endswith('.pt') and file.startswith('epoch-{0}+'.format(best_epoch)):
        weights_path = os.path.join(experiment_dir, file)
weights = torch.load(weights_path)
if gpu_available:
    weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
else:
    weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()]) 
try:
    model.load_state_dict(weights)
except:
    state = model.state_dict()
    diff = list(set(state.keys()).difference(set(weights.keys())))
    state.update(weights)
    model.load_state_dict(state)
        
# Initiate evaluation mode
model.eval()



all_preds = []
# rand_input = (200 - 1) * torch.rand(30, 6, 29, 29) + 1
rand_input = (0.001 - 0) * torch.rand(30, 6, 29, 29)
print(rand_input)
for batch_id, data in enumerate(rand_input):
    data = data.unsqueeze(0)
    with torch.no_grad():
        if gpu_available:
            data = Variable(data.float().cuda(output_device), requires_grad=False, volatile=True)
        else:
            data = Variable(data.float(), requires_grad=False, volatile=True)
        """ outputs = model.forward(data) """
        output, feature = model(data)
        print(f"Output: {output}")
        all_preds.append(output.data.cpu().numpy())
        print(all_preds)

preds_folds = []
preds = np.concatenate(all_preds)
softmax = np.zeros(preds.shape)
for i in range(preds.shape[0]):
    p = preds[i,:]
    softmax[i,:] = np.exp(p) / np.sum(np.exp(p), axis=0)
            
preds_folds.append(softmax)
preds_folds = np.asarray(preds_folds)

# Compute risk of outcome from ensemble of cross-validation folds
ensemble_preds = np.median(preds_folds, axis=0)
print(ensemble_preds)
print("------")
outcome_risk = np.median(ensemble_preds, axis=0)[1]

prediction_threshold = 0.5

# Perform classification of outcome
outcome = True if outcome_risk >= prediction_threshold else False

# Determine classification uncertainty
outcome_risk_folds = np.median(preds_folds, axis=1)[...,1]
lower_quartile = np.percentile(outcome_risk_folds, 25)
upper_quartile = np.percentile(outcome_risk_folds, 75)
if outcome and lower_quartile > prediction_threshold:
    certain = True
elif outcome:
    certain = False
elif not outcome and upper_quartile < prediction_threshold:
    certain = True
else:
    certain = False

results = {'risk': outcome_risk, 'prediction_threshold': prediction_threshold, 'positive_classification': outcome, 'certain_classification': certain}
print(results)