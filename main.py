import os, sys

""" INSERT HYPERPARAMETER SETTINGS FOR NEURAL ARCHITECTURE SEARCH, CROSS-VALIDATION, AND EVALUATION """
""" Project """

#### Step 1: Create a new folder under 'projects' and enter the name of the folder below. Create a 'data' and 'experiments' subfolder. 
# NOTE: You do not have to create a new project for each experiment. Your experiments subfolder will contain all your experiments  
project_name = 'im2021' # <--- Enter the name of your project folder

project_dir = os.path.join('projects', project_name)
sys.path.append(project_dir)


""" Search details """

# Options

#### Step 2: Enter the name of your neural architecture search
# NOTE: Remember to give your search a unique name that is not contained in your 'experiments' subfolder
search_name = '21092022 1522 IM2021' # <--- Enter the name of your search

#### Step 3: Decide if you want to run neural architecture search (NAS), cross-validation, or model evaluation. The NAS procedure 'search' obtains the model with the highest Area Under the ROC Curve (AUC) on the main validation dataset. The cross-validation procedure 'cross_val' obtains multiple model instances (with different sets of weights) through training and validation on different subsets of data (i.e., k-fold cross-validation). The evaluation procedure 'evaluate' estimates likelihood of movement outcome on test data and aggregates the prediction across the ensemble of model instances to perform binary classification of movement outcome. 
search = True # <-- Assign [True, False] 
crossval = True # <-- Assign [True, False] 
evaluate = True # <-- Assign [True, False] 

#### Step 4: Choose usage of output device and number of workers
output_device = 0 # <-- Assign device with CUDA to use GPU
num_workers = 4

#### Step 5: Choose model type and configuration
model_script = 'models.gcn_search_model' # <-- Assign model script (e.g., models.gcn_search_model)
input_dimensions = 2 # <-- Assign number of dimensions in the coordinate space (e.g., 2)
input_spatial_resolution = 19 # <-- Assign number of joints in infant skeleton (e.g., 19)
input_temporal_resolution = 150 # <-- Assign number of time steps (i.e., frames) in a window (e.g., 150)
num_input_branches = 3
edge_importance_weighting = True
dropout = 0

#### Step 6: Choose graph type and configuration.
body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'] # <-- Assign body parts in skeleton (e.g., In-Motion 19: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'], In-Motion 29: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe', 'left_big_toe'])
neighbor_link = [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5, 8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14)] # <-- Assign neighboring body parts that are connected by bones in the skeleton (e.g., In-Motion 19: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5, 8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14)], In-Motion 29: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5,8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14), (19,7), (20,7), (21,11), (22,11), (23,15), (24,15), (25,15), (26,18), (27,18), (28,18)])
center = 8 # <-- Assign index of body part in center of skeleton (e.g., In-Motion 19/29: 8 for thorax)
bone_conns = [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17] # <-- Assign parent (i.e., body part closer to the center) of each body part (e.g., In-Motion 19: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17], In-Motion 29: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18, 18])
thorax_index = 8 # <-- Assign index of thorax bpdy part (e.g., In-Motion 19/29: 8)
pelvis_index = 12 # <-- Assign index of pelvis bpdy part (e.g., In-Motion 19/29: 12)
use_mask = True 

#### Step 7: Choose type of input features
absolute = True # <-- Assign [True, False] 
relative = False # <-- Assign [True, False] 
motion1 = True # <-- Assign [True, False] 
motion2 = False # <-- Assign [True, False]
bone = True # <-- Assign [True, False] 
bone_angle = False # <-- Assign [True, False]

#### Step 8: Set frames per second of coordinate files and median filter stride for preprocessing.
frames_per_second = 30.0 # <-- Assign (i.e., assumes consistent frame rate across coordinate files)
filter_stride = 2

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
k = 5
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
search_train_dataset = 'train1'
search_val_dataset = 'val1'
search_num_epochs = 100
search_save_interval = 20000000
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


""" START SEARCH SCRIPT (DO NOT CHANGE)"""
""" Dependencies """

# External dependencies
import json
import torch
import numpy as np

# Local dependencies
model = __import__(model_script, fromlist=['object'])
from utils import process
from utils.search import print_status, update_best, get_candidate, update_search_space
from utils.trainval import trainval
from utils.test import test
from utils.evaluate import evaluate as eval


""" Initialize search directory """
search_dir = os.path.join(project_dir, 'searches', search_name)
experiments_dir = os.path.join(search_dir, 'experiments')
os.makedirs(search_dir, exist_ok=True)
os.makedirs(experiments_dir, exist_ok=True)
processed_data_dir = os.path.join(project_dir, 'data', 'processed')


""" Store experiment hyperparameters """

# Construct dictionary of hyperparameters
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

# Store hyperparameters as JSON file
with open(os.path.join(search_dir, 'hyperparameters.json'), 'w') as json_file:  
    json.dump(hyperparameters, json_file)


""" Initialize data """

# Process coordinate files and outcomes (assuming raw folder exists with one coordinate CSV and associated row in outcome CSV per individual)
process.process(project_dir, processed_data_dir, test_size=test_size, crossval_folds=crossval_folds, num_dimensions=input_dimensions, num_joints=input_spatial_resolution, filter_stride=filter_stride)


""" Neural architecture search """

if search:
    
    # Initialize search history
    candidate_history = []
    best_performance = 0.0
    best_candidate_num = None
    best = None
    
    # Initialize search space with uniform probabilities
    for choice in search_space.keys():
        alternatives = search_space[choice]
        num_alternatives = len(alternatives)
        search_space[choice] = {}
        for alternative in alternatives:
            search_space[choice][alternative] = 1/num_alternatives
    
    # Obtain initial population with random search
    random_search = True
    random_candidate_num = 1
    population = []
    while len(population) < k:

        # Fetch unexplored candidate
        candidate, candidate_string, candidate_history = get_candidate(search_space, candidate_history)
        if candidate is None:
            break

        # Perform training and validation
        performance = trainval(processed_data_dir, experiments_dir, 'r' + str(random_candidate_num), candidate, hyperparameters, crossval_fold=None)

        # Keep track of best candidate
        best, best_performance, best_candidate_num = update_best(candidate_string, performance, 'r{0}'.format(random_candidate_num), best, best_performance, best_candidate_num)

        # Include in population if meets performance requirement
        if performance >= performance_threshold:
            population.append((candidate_string, performance, 'r' + str(random_candidate_num)))
            population = sorted(population, key=lambda x: x[1])
        
        # Print search status
        print_status(candidate_string, performance, 'r{0}'.format(random_candidate_num), best, best_performance, best_candidate_num, population)

        random_candidate_num += 1

    # Update population with search for K best candidates
    if len(population) == k:
            
        # Sort population on performance
        population = sorted(population, key=lambda x: x[1])

        # Initialize temperature
        temperature = start_temperature

        # Update search space from candidates in population  
        search_space = update_search_space(population, temperature=temperature)

        # Perform search
        candidate_num = 1
        unsuccessful_trials = 0
        while True:

            # Fetch unexplored candidate
            candidate, candidate_string, candidate_history = get_candidate(search_space, candidate_history)
            if candidate is None:
                break

            # Perform training and validation
            performance = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=None)

            # Keep track of best candidate
            best, best_performance, best_candidate_num = update_best(candidate_string, performance, candidate_num, best, best_performance, best_candidate_num)

            # Update population if improves upon lowest performing candidate in population
            if performance > population[0][1]:
                population[0] = (candidate_string, performance, candidate_num)
                population = sorted(population, key=lambda x: x[1])
                unsuccessful_trials = 0
            else:
                unsuccessful_trials += 1

            # Update search space probabilities from population
            search_space = update_search_space(population, temperature=temperature)

            # Decrease search temperature or terminate search if reached end temperature
            if unsuccessful_trials == k and temperature == end_temperature:
                print_status(candidate_string, performance, candidate_num, best, best_performance, best_candidate_num, population, temperature, unsuccessful_trials)
                break
            elif unsuccessful_trials == k:
                temperature -= temperature_drop
                unsuccessful_trials = 0
                
            # Print search status
            print_status(candidate_string, performance, candidate_num, best, best_performance, best_candidate_num, population, temperature, unsuccessful_trials)
                
            candidate_num += 1

    
""" Cross-validation """

# Fetch candidate
if crossval or evaluate:
    if search:
        _, candidate_performance, candidate_num = population[-1]
    else:
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

# Perform cross-validation of candidate
if crossval:    
                    
    # Iterate over cross-validation folds
    for crossval_fold in range(1, crossval_folds+1):
    
        # Perform training and validation
        crossval_fold_performance = trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=crossval_fold)
    
    
""" Evaluation """

if evaluate:  
    
    # Iterate over cross-validation folds
    preds_folds = []
    for crossval_fold in range(1, crossval_folds+1):
    
        # Perform testing
        preds, labels, video_ids = test(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold=crossval_fold)
        softmax = np.zeros(preds.shape)
        for i in range(preds.shape[0]):
            softmax[i,:] = np.exp(preds[i,:]) / np.sum(np.exp(preds[i,:]), axis=0)
        preds_folds.append(softmax)
    preds_folds = np.asarray(preds_folds)
        
    # Aggregate predictions across ensemble of cross-validation folds
    preds_ensemble = np.median(preds_folds, axis=0)   
    
    # Store ensemble predictions
    preds_object = []
    for video_id, pred, label in zip(video_ids, preds_ensemble, labels):
        preds_object.append((video_id, pred, label))
    with open(os.path.join(self.search_dir, 'ensemble_test_preds.pkl'), 'wb') as f:
        pickle.dump(preds_object, f)
    
    # Compute Area Under ROC Curve
    ensemble_auc, ensemble_accuracy, ensemble_f1, ensemble_sensitivity, ensemble_specificity, ensemble_ppv, ensemble_npv, ensemble_balanced_accuracy = eval(preds_ensemble, labels, video_ids, aggregate_binary=hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=hyperparameters['evaluation']['median_aggregation'], prediction_threshold=hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=True, subject=True, normalized=True)
    ensemble_window_auc, ensemble_window_accuracy, ensemble_window_f1, ensemble_window_sensitivity, ensemble_window_specificity, ensemble_window_ppv, ensemble_window_npv, ensemble_window_balanced_accuracy = eval(preds_ensemble, labels, video_ids, aggregate_binary=hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=hyperparameters['evaluation']['median_aggregation'], prediction_threshold=hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=True, subject=False, normalized=True)
    
    # Store ensemble test results as JSON file
    ensemble_test_results = {}
    ensemble_test_results['ensemble_test_auc'] = ensemble_auc
    ensemble_test_results['ensemble_test_accuracy'] = ensemble_accuracy
    ensemble_test_results['ensemble_test_f1'] = ensemble_f1
    ensemble_test_results['ensemble_test_sensitivity'] = ensemble_sensitivity
    ensemble_test_results['ensemble_test_specificity'] = ensemble_specificity
    ensemble_test_results['ensemble_test_ppv'] = ensemble_ppv
    ensemble_test_results['ensemble_test_npv'] = ensemble_npv
    ensemble_test_results['ensemble_test_balanced_accuracy'] = ensemble_balanced_accuracy
    ensemble_test_results['ensemble_test_window_auc'] = ensemble_window_auc
    ensemble_test_results['ensemble_test_window_accuracy'] = ensemble_window_accuracy
    ensemble_test_results['ensemble_test_window_f1'] = ensemble_window_f1
    ensemble_test_results['ensemble_test_window_sensitivity'] = ensemble_window_sensitivity
    ensemble_test_results['ensemble_test_window_specificity'] = ensemble_window_specificity
    ensemble_test_results['ensemble_test_window_ppv'] = ensemble_window_ppv
    ensemble_test_results['ensemble_test_window_npv'] = ensemble_window_npv
    ensemble_test_results['ensemble_test_window_balanced_accuracy'] = ensemble_window_balanced_accuracy
    with open(os.path.join(search_dir, 'ensemble_test_results.json'), 'w') as json_file:  
        json.dump(ensemble_test_results, json_file)
