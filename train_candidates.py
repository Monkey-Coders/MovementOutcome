import os, sys
from utils.trainval import trainval
import torch
from datetime import datetime
import json
import io
import time

from utils_functions import get_model_from_candidate_file, get_candidate
RUN = datetime.now()

search_name = RUN.strftime("%Y/%B/%d/%H:%M:%S") # <--- Enter the name of your search

project_name = 'sp2022' # <--- Enter the name of your project folder
body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe', 'left_big_toe']
neighbor_link = [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5,8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14), (19,7), (20,7), (21,11), (22,11), (23,15), (24,15), (25,15), (26,18), (27,18), (28,18)]
center = 8 # <-- Assign index of body part in center of skeleton (e.g., In-Motion 19/29: 8 for thorax)
bone_conns = [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18, 18]
thorax_index = 8 # <-- Assign index of thorax body part (e.g., In-Motion 19/29: 8)
pelvis_index = 12 # <-- Assign index of pelvis body part (e.g., In-Motion 19/29: 12)
use_mask = True 

absolute = True # <-- Assign [True, False] 
relative = False # <-- Assign [True, False] 
motion1 = True # <-- Assign [True, False] 
motion2 = False # <-- Assign [True, False]
bone = True # <-- Assign [True, False] 
bone_angle = False # <-- Assign [True, False]

project_dir = os.path.join('projects', project_name)
sys.path.append(project_dir)

#### Step 4: Choose usage of output device and number of workers
output_device = 0 # <-- Assign device with CUDA to use GPU
num_workers = 4

#### Step 8: Set frames per second of coordinate files and median filter stride for preprocessing.
frames_per_second = 30.0 # <-- Assign (i.e., assumes consistent frame rate across coordinate files)
filter_stride = 2 # <-- Assign stride of median filter for temporal smoothing of coordinates.

#### Step 9: Set hyperparameters for training and validation (e.g., mini batch size and loss filter size)
trainval_batch_size = 32 
loss_filter_size = 5

#### Step 10: Balance the training set by adjusting the number of positive samples per negative sample (e.g., for In-Motion with around 5 times more negative samples than positive samples (15% prevalence of CP), each positive sample is represented 5 times more frequent than each negative sample).
train_num_positive_samples_per_negative_sample = 5 # <-- Assign in accordance with prevalence of outcome in training set
train_num_per_positive_sample = train_num_positive_samples_per_negative_sample*12
train_num_per_negative_sample = 12

#### Step 11: Set hyperparameters for the optimizer (SGD) 
learning_rate = 0.0005
momentum = 0.9
weight_decay = 0.0
nesterov = True 
reduction_factor = 0.0
steps = []
print_log = True 
seed = 1

#### Step 12: Set hyperparameters for the data augmentation (e.g., random start frame and random perturbations with rotation/scaling/translation)
random_start = True 
random_perturbation = True 
max_angle_candidate = 45
min_scale_candidate = 0.7
max_scale_candidate = 1.3
max_translation_candidate = 0.3
standardize_rotation = True
roll_sequence = True

#### Step 13: Set evaluation options (e.g., test set size (defaults to 25% of all individuals), mini batch size, number of frames between each sample and aggregation scheme)
test_size = 0.25
evaluation_batch_size = trainval_batch_size
parts_distance = 75
aggregate_binary = False
aggregate_binary_threshold = None
median_aggregation = True
prediction_threshold = 0.5
evaluation_save_preds = True

#### Step 14: Set neural architecture search specific hyperparameters
k = 2
start_temperature = 10
end_temperature = 1
temperature_drop = 3
performance_threshold = 0.9
search_space = {
    'graph': ['spatial', 'dis2', 'dis4', 'dis4+2'],
    'input_width': [6, 8, 10, 12],
    'num_input_modules': [1, 2, 3],
    'initial_block_type': ['basic', 'bottleneck', 'mbconv'],
    'initial_residual': ['null', 'block', 'module', 'dense'],
    'input_temporal_scales': [1, 2, 3, 'linear'],
    'initial_main_width': [6, 8, 10, 12],
    'num_main_levels': [1, 2],
    'num_main_level_modules': [1, 2, 3],
    'block_type': ['basic', 'bottleneck', 'mbconv'],
    'bottleneck_factor': [2, 4],
    'residual': ['null', 'block', 'module', 'dense'],
    'main_temporal_scales': [1, 2, 3, 'linear'],
    'temporal_kernel_size': [3, 5, 7, 9],
    'se': ['null', 'inner', 'outer', 'both'],
    'se_ratio': [2, 4],
    'se_type': ['relative', 'absolute'],
    'nonlinearity': ['relu', 'swish'],
    'attention': ['null', 'channel', 'frame', 'joint'],
    'pool': ['global', 'spatial']
}
search_train_dataset = 'train'
search_val_dataset = 'val'
search_num_epochs = 100
search_save_interval = 5
search_save_preds = False 
search_critical_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
search_critical_epoch_values = [0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95]

#### Step 15: Set hyperparameters for cross-validation
crossval_folds = 7
crossval_train_datasets = ['train{0}'.format(n) for n in range(1, crossval_folds+1)]
crossval_val_datasets = ['val{0}'.format(n) for n in range(1, crossval_folds+1)]
crossval_num_epochs = 200
crossval_save_interval = 1
crossval_save_preds = False
crossval_critical_epochs = []
crossval_critical_epoch_values = []

#### Step 5: Choose model type and configuration
model_script = 'models.gcn_search_model' # <-- Assign model script (e.g., models.gcn_search_model)
input_dimensions = 2 # <-- Assign number of dimensions in the coordinate space (e.g., 2)
input_spatial_resolution = 29 # <-- Assign number of joints in human skeleton (e.g., 19)
input_temporal_resolution = 150 # <-- Assign number of time steps (i.e., frames) in a window (e.g., 150)
num_input_branches = 3
edge_importance_weighting = True
dropout = 0
hyperparameters = {'devices': {'output_device': output_device,
                                'gpu_available': torch.cuda.is_available(),
                                'num_workers': num_workers},
                       'model': {'model_script': model_script,
                                'input_dimensions': input_dimensions,
                                'input_spatial_resolution': input_spatial_resolution,
                                'input_temporal_resolution': input_temporal_resolution,
                                'num_input_branches': num_input_branches,
                                'edge_importance_weighting': edge_importance_weighting,
                                'dropout': dropout},
                      'graph': {'body_parts': body_parts,
                                'neighbor_link': neighbor_link,
                                'center': center,
                                'bone_conns': bone_conns,
                                'thorax_index': thorax_index,
                                'pelvis_index': pelvis_index,
                                'use_mask': use_mask},
                      'features': {'absolute': absolute,
                                'relative': relative,
                                'motion1': motion1,
                                'motion2': motion2,
                                'bone': bone,
                                'bone_angle': bone_angle},
                      'preprocessing': {'frames_per_second': frames_per_second,
                                'filter_stride': filter_stride},
                      'trainval': {'trainval_batch_size': trainval_batch_size,
                                'loss_filter_size': loss_filter_size}, 
                      'balance': {'train_num_positive_samples_per_negative_sample': train_num_positive_samples_per_negative_sample,
                                'train_num_per_positive_sample': train_num_per_positive_sample,
                                'train_num_per_negative_sample': train_num_per_negative_sample},
                      'optimizer': {'learning_rate': learning_rate,
                                'momentum': momentum,
                                'weight_decay': weight_decay,
                                'nesterov': nesterov,
                                'reduction_factor': reduction_factor,
                                'steps': steps,
                                'print_log': print_log,
                                'seed': seed},
                      'augmentation': {'random_start': random_start,
                                'random_perturbation': random_perturbation,
                                'max_angle_candidate': max_angle_candidate, 
                                'min_scale_candidate': min_scale_candidate,
                                'max_scale_candidate': max_scale_candidate,
                                'max_translation_candidate': max_translation_candidate,
                                'standardize_rotation': standardize_rotation,
                                'roll_sequence': roll_sequence},
                      'evaluation': {'test_size': test_size,
                                'evaluation_batch_size': evaluation_batch_size,
                                'parts_distance': parts_distance,
                                'aggregate_binary': aggregate_binary,
                                'aggregate_binary_threshold': aggregate_binary_threshold,
                                'median_aggregation': median_aggregation,
                                'prediction_threshold': prediction_threshold,
                                'evaluation_save_preds': evaluation_save_preds},
                      'search': {'k': k,
                                'start_temperature': start_temperature,
                                'end_temperature': end_temperature,
                                'temperature_drop': temperature_drop,
                                'performance_threshold': performance_threshold,
                                'search_space': search_space, 
                                'search_train_dataset': search_train_dataset,
                                'search_val_dataset': search_val_dataset,
                                'search_num_epochs': search_num_epochs,
                                'search_save_interval': search_save_interval,
                                'search_save_preds': search_save_preds,
                                'search_critical_epochs': search_critical_epochs,
                                'search_critical_epoch_values': search_critical_epoch_values},
                      'crossval': {'crossval_folds': crossval_folds,
                                'crossval_train_datasets': crossval_train_datasets,
                                'crossval_val_datasets': crossval_val_datasets,
                                'crossval_num_epochs': crossval_num_epochs,
                                'crossval_save_interval': crossval_save_interval,
                                'crossval_save_preds': crossval_save_preds,
                                'crossval_critical_epochs': crossval_critical_epochs,
                                'crossval_critical_epoch_values': crossval_critical_epoch_values}
                      }


processed_data_dir = os.path.join(os.environ["HOME"], "..", "..", "data", "stud", "maxts", "data", "dim", "processed", "infants_resampled_smoothed")
search_dir = os.path.join(project_dir, 'searches', search_name)
experiments_dir = os.path.join(search_dir, 'experiments')
base_path = "projects/sp2022/searches/2022"

zero_cost_folder = "zero_cost_experiments"
zero_cost_experiments_folder_exists = os.path.exists(zero_cost_folder)
if not zero_cost_experiments_folder_exists:
    os.makedirs(zero_cost_folder)


PATH = f"{zero_cost_folder}/results.json"
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    # checks if file exists
    print ("File exists and is readable")
else:
    print ("Either file is missing or is not readable, creating file...")
    with io.open(PATH, 'w') as file:
        file.write(json.dumps({}))

with open(PATH) as f:
    candidate_dict = json.load(f)

counter = 0
for candidate_key, values in candidate_dict.items():
    candidate_num = counter
    candidate = eval(candidate_key)

    results = candidate_dict[candidate_key]
    """candidate_needs_synflow = "synflow" not in results
    
    if candidate_needs_synflow:
        score, _ = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=False, zero_cost_method = "synflow")
        results["synflow"] = score
    """

    candidate_needs_snip = "snip" not in results
    if candidate_needs_snip:
        score, _ = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=False, zero_cost_method = "snip")
        results["snip"] = score

    print(results)
    exit()
    candidate_needs_grad_norm = "grad_norm" not in results
    if candidate_needs_grad_norm:
        score, _ = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=False, zero_cost_method = "grad_norm")
        results["grad_norm"] = score

    candidate_needs_to_train = "best_auc" not in results
    if candidate_needs_to_train:
        print("="*100)
        print(f"Training candidate: {counter}")
        performance, validation_results = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=True, use_zero_cost=False)
        results["val_accuracy"] = performance
        results["num_parameters"] = validation_results["num_parameters"]
        results["num_flops"] = validation_results["num_flops"]
        results["best_loss"] = validation_results["best_loss"]
        results["best_auc"] = validation_results["best_auc"]
        print(f"Performance: {performance}")
        print("="*100)
        print(f"Finished training candidate: {counter}")
        print("="*100)
    

    candidate_dict[candidate_key] = results
    print("Writing results to file")
    with open(PATH, "w") as f:
        json.dump(candidate_dict, f)
    counter += 1


exit()


for month in os.listdir(base_path):
    for date in os.listdir(f"{base_path}/{month}"):
        for search in os.listdir(f"{base_path}/{month}/{date}"):
            counter = 1
            for experiment in os.listdir(f"{base_path}/{month}/{date}/{search}/experiments"):
                print(f"{base_path}/{month}/{date}/{search}/experiments/{experiment}/")
                candidate = get_candidate(f"{base_path}/{month}/{date}/{search}/experiments/{experiment}/")
                if candidate is None:
                    continue
                candidate_stringified = str(candidate)
                candidate_num = 0
                if candidate_stringified in candidate_dict:
                    results = candidate_dict[candidate_stringified]
                else:
                    results = {}
                
                    # Skip those we have not trained for now
                    continue

                candidate_needs_synflow = "synflow" not in results
                if candidate_needs_synflow:
                    score, _ = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=False, zero_cost_method = "synflow")
                    results["synflow"] = score

                candidate_needs_grad_norm = "grad_norm" not in results
                if candidate_needs_grad_norm:
                    score, _ = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=False, zero_cost_method = "grad_norm")
                    results["grad_norm"] = score

                candidate_needs_to_train = "val_accuracy" not in results
                if candidate_needs_to_train:
                    print("="*100)
                    print(f"Training candidate: {counter}")
                    print(counter)
                    print(candidate_stringified)
                    performance, validation_results = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None, train=True, use_zero_cost=False)
                    results["val_accuracy"] = performance
                    results["num_parameters"] = validation_results["num_parameters"]
                    results["num_flops"] = validation_results["num_flops"]
                    results["best_loss"] = validation_results["best_loss"]
                    results["best_auc"] = validation_results["best_auc"]
                    print(f"Performance: {performance}")
                    print("="*100)
                    print(f"Finished training candidate: {counter}")
                    print("="*100)
                

                candidate_dict[candidate_stringified] = results
                print("Writing results to file")
                with open(PATH, "w") as f:
                    json.dump(candidate_dict, f)
                counter += 1



            
with open(PATH, "w") as f:
    json.dump(candidate_dict, f)
                

