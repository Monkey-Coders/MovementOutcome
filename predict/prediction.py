import os, sys
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import random


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    

class EvalFeeder(Dataset):
    def __init__(self, data, graph, input_temporal_resolution=150, parts_distance=75, standardize_rotation=True, absolute=True, relative=False, motion1=False, motion2=False, bone=False, bone_angle=False):
        self.graph = graph
        self.input_temporal_resolution = input_temporal_resolution
        self.parts_distance = parts_distance
        self.standardize_rotation = standardize_rotation

        self.init_data = data
        self.load_data()

        # Which features to use
        self.absolute = absolute
        self.relative = relative
        self.motion1 = motion1
        self.motion2 = motion2
        self.bone = bone
        self.bone_angle = bone_angle
        
        from utils.helpers import create_bone_motion_features, rotate, get_rotation_angle
        self.create_bone_motion_features = create_bone_motion_features
        self.rotate = rotate
        self.get_rotation_angle = get_rotation_angle

    def load_data(self):

        # Extract sequence parts
        data = []
        
        sample_data = self.init_data[0,...]
        sample_num_frames = sample_data.shape[1]

        # Construct sequence parts of self.input_temporal_resolution length and self.parts_distance distance to consecutive parts
        for start_frame in range(0, sample_num_frames, self.parts_distance):
            if start_frame > sample_num_frames - self.input_temporal_resolution:
                data.append(sample_data[:, sample_num_frames - self.input_temporal_resolution:sample_num_frames, :])
                break
            else:
                data.append(sample_data[:, start_frame:start_frame+self.input_temporal_resolution, :])

        self.data = np.array(data)

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return self
    
    def get_channels(self):
        data = self.__getitem__(0)
        in_channels, _, _ = data.shape
        return in_channels

    def __getitem__(self, index):
        
        # Fetch part information
        sample_data = np.array(self.data[index])

        # Rotate the body to start the sequence with a vertical spine
        if self.standardize_rotation:
            angle = self.get_rotation_angle(sample_data, self.graph)
            sample_data = self.rotate(sample_data, angle)

        C, T, V = sample_data.shape
        sample_data = np.reshape(sample_data, (C, T, V, 1))
        sample_data = self.create_bone_motion_features(sample_data, self.graph.bone_conns)

        # Concatenate the new features into the existing format and remove M again
        F, C, T, V, M = sample_data.shape
        sample_data = np.reshape(sample_data, (F*C, T, V))

        keep_features = []
        if self.absolute:
            keep_features += [0,1]
        if self.relative:
            keep_features += [2,3]
        if self.motion1:
            keep_features += [4,5]
        if self.motion2:
            keep_features += [6,7]
        if self.bone:
            keep_features += [8,9]
        if self.bone_angle:
            keep_features += [10,11]
        
        sample_data = sample_data[keep_features,:,:]
        return sample_data

def predict(coords_dir, coords_path):
    
    """ Project """
    
    #### Step 1: Project name
    project_name = 'im2021' # <---Enter the name of your project folder
    
    project_dir = os.path.join('../projects', project_name)
    sys.path.append(project_dir)
    sys.path.append(os.path.join('..'))
    
    
    """ Search details """

    #### Step 2: Search name
    search_name = '21092022 1522 IM2021' # <--- Enter the name of your search

    #### Step 3: Decide if you want to save predicted risk of outcome, classification and associated uncertainty in CSV file and/or visualize body keypoints of highest contribution towards risk (i.e., class activation mapping)
    save = True # <-- Assign  [True, False] 
    visualize = True # <-- Assign  [True, False] 
    
    #### Step 4: Choose usage of output device and number of workers
    output_device = 0 # <-- Assign device with CUDA to use GPU
    num_workers = 4

    #### Step 5: Choose model type and configuration
    model_script = 'models.gcn_search_model' # <-- Assign model script (e.g., models.gcn_search_model)
    input_dimensions = 2 # <-- Assign number of dimensions in the coordinate space (e.g., 2)
    input_spatial_resolution = 19 # <-- Assign number of joints in human skeleton (e.g., 19)
    input_temporal_resolution = 150 # <-- Assign number of time steps (i.e., frames) in a window (e.g., 150)
    num_input_branches = 3
    edge_importance_weighting = True
    dropout = 0

    #### Step 6: Choose graph type and configuration.
    body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'] # <-- Assign body parts in skeleton (e.g., In-Motion 19: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'], In-Motion 29: ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe', 'left_big_toe'])
    neighbor_link = [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5, 8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14)] # <-- Assign neighboring body parts that are connected by bones in the skeleton (e.g., In-Motion 19: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5, 8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14)], In-Motion 29: [(0,1), (2,1), (3,1), (1,4), (9,8), (10,9), (11,10), (5,8), (6,5), (7,6), (4,8), (12,8), (16,12), (17,16), (18,17), (13,12), (14,13), (15,14), (19,7), (20,7), (21,11), (22,11), (23,15), (24,15), (25,15), (26,18), (27,18), (28,18)])
    center = 8 # <-- Assign index of body part in center of skeleton (e.g., In-Motion 19/29: 8 for thorax)
    bone_conns = [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17] # <-- Assign parent (i.e., body part closer to the center) of each body part (e.g., In-Motion 19: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17], In-Motion 29: [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18, 18])
    thorax_index = 8 # <-- Assign index of thorax body part (e.g., In-Motion 19/29: 8)
    pelvis_index = 12 # <-- Assign index of pelvis body part (e.g., In-Motion 19/29: 12)
    use_mask = True 
    sample_coords = [(0.51, 0.07), (0.51, 0.15), (0.473, 0.187), (0.55, 0.183), (0.51, 0.22), (0.47, 0.26), (0.425, 0.28), (0.383, 0.254), (0.515, 0.262), (0.565, 0.265), (0.605, 0.265), (0.637, 0.252),(0.508, 0.43), (0.47, 0.42), (0.453, 0.562), (0.465, 0.70), (0.546, 0.44), (0.532, 0.585), (0.50, 0.72)] # <-- Assign dummy coordinates for each body part

    #### Step 7: Choose type of input features
    absolute = True # <-- Assign [True, False] 
    relative = False # <-- Assign [True, False] 
    motion1 = True # <-- Assign [True, False] 
    motion2 = False # <-- Assign [True, False]
    bone = True # <-- Assign [True, False] 
    bone_angle = False # <-- Assign [True, False]

    #### Step 8: Set frames per second of coordinate files and median filter stride for preprocessing.
    frames_per_second = 30.0 # <-- Assign (i.e., assumes consistent frame rate across coordinate files)
    filter_stride = 2 # <-- Assign stride of median filter for temporal smoothing of coordinates.
    
    #### Step 10: Set evaluation options (e.g., mini batch size, number of frames between each sample, and aggregation scheme)
    evaluation_batch_size = 32
    parts_distance = 75
    median_aggregation = True
    prediction_threshold = 0.5
    seed = 1
    standardize_rotation = True
    
    #### Step 11: Set hyperparameters for cross-validation
    crossval_folds = 7


    """ Dependencies """

    # External dependencies
    import json
    import csv
    import math
    from collections import OrderedDict
    from torch.autograd import Variable
    import matplotlib.pyplot as plt

    # Local dependencies
    m = __import__(model_script, fromlist=['object'])
    from utils.helpers import median_filter, coords_raw_to_norm
    from utils import graph
    

    """ Define directories """
    
    search_dir = os.path.join(project_dir, 'searches', search_name)
    experiments_dir = os.path.join(search_dir, 'experiments')
    results_dir = os.path.join(search_dir, coords_dir.split('/')[-1])
    os.makedirs(results_dir, exist_ok=True)
    
    """ Obtain skeleton sequence """
        
    # Calculate median trunk length and median pelvis (assumes body keypoints of thorax and pelvis exist)
    trunk_lengths = []
    pelvis_xs = []
    pelvis_ys = []
    frames = {}
    with open(os.path.join(coords_path), newline='') as individual_file:
        individual_reader = csv.reader(individual_file, delimiter=',', quotechar='|')
        header = next(individual_reader)
        for row in individual_reader:
            if len(row) > 0:
                frames[int(row[0])] = np.swapaxes(np.asarray([[row[body_part_index], row[body_part_index+1]] for body_part_index in range(1, len(row), 2)], dtype=np.float32), 0, 1)
                thorax_x = float(row[header.index('thorax_x')])
                thorax_y = float(row[header.index('thorax_y')])
                pelvis_x = float(row[header.index('pelvis_x')])
                pelvis_y = float(row[header.index('pelvis_y')])
                trunk_length = math.sqrt((thorax_x - pelvis_x)**2 + (thorax_y - pelvis_y)**2)
                trunk_lengths.append(trunk_length)
                pelvis_xs.append(pelvis_x)
                pelvis_ys.append(pelvis_y)            
    median_trunk_length = np.median(trunk_lengths)
    median_pelvis_x = np.median(pelvis_xs)
    median_pelvis_y = np.median(pelvis_ys)

    # Filter, centralize and normalize coordinates 
    num_frames = len(trunk_lengths)
    individual_sequence = np.zeros((1, input_dimensions, num_frames, input_spatial_resolution))
    for frame in frames.keys():

        # Apply median filter
        filter_frame_coords = median_filter(frames, frame, num_frames, filter_stride)

        # Centralize and normalize
        norm_frame_coords = coords_raw_to_norm(filter_frame_coords, median_pelvis_x, median_pelvis_y, median_trunk_length)
        individual_sequence[0, :, frame-1, :] = norm_frame_coords

    
    """ Obtain architecture of best performing candidate from K-Best Search """
    
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
    
    
    """ Prediction """
    
    # Iterate over cross-validation folds
    preds_folds = []
    cams_folds = []
    for crossval_fold in range(1, crossval_folds+1):
        
        # Determine experiment dir
        experiment_dir = os.path.join(experiments_dir, 'crossval_{0}_val{1}'.format(candidate_num, crossval_fold))
        
        # Initialize graph
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

        # Initialize feeder
        individual_feeder = EvalFeeder(data=individual_sequence, graph=graph_input, standardize_rotation=standardize_rotation, input_temporal_resolution=input_temporal_resolution, parts_distance=parts_distance, absolute=absolute, relative=relative, motion1=motion1, motion2=motion2, bone=bone, bone_angle=bone_angle)
        
        # Obtain data loader
        individual_dataloader = torch.utils.data.DataLoader(dataset=individual_feeder, batch_size=evaluation_batch_size, shuffle=False, num_workers=num_workers, drop_last=False, worker_init_fn=init_seed(seed))
        
        # Determine number of input channels
        input_channels = individual_feeder.get_channels()
        
        # Initialize model
        
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
        output_device = int(output_device)
        gpu_available = torch.cuda.is_available()
        
        # Initialize model
        model = m.Model(num_classes=2, graphs=[graph_input, graph_main] if len(labeling_mode) == 2 else [graph_input], input_channels=input_channels, edge_importance_weighting=edge_importance_weighting, dropout=dropout, num_input_branches=num_input_branches, attention=attention, se_outer=se_outer, se_inner=se_inner, initial_residual=initial_residual, residual=residual, initial_block_type=initial_block_type, block_type=block_type, input_width=input_width, initial_main_width=initial_main_width, temporal_kernel_size=temporal_kernel_size, num_input_modules=num_input_modules, num_main_levels=num_main_levels, num_main_level_modules=num_main_level_modules, input_temporal_scales=input_temporal_scales, main_temporal_scales=main_temporal_scales, bottleneck_factor=bottleneck_factor, se_ratio=se_ratio, relative_se=relative_se, swish_nonlinearity=swish_nonlinearity, spatial_pool=spatial_pool)
        if gpu_available:
            model = model.cuda(output_device)
        
        # Fetch epoch with highest performance
        with open(os.path.join(experiment_dir, 'validation_results.json'), 'r') as json_file: 
            best_epoch = json.load(json_file)['best_auc_epoch']
            
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
        
        # Prepare for class activation mapping
        if visualize:
            try:
                target_layer = model.st_gcn_main[-1].attention.nonlinearity
            except:
                target_layer = model.st_gcn_main[-1].tcn.nonlinearity
        
        # Fetch data
        process = tqdm(individual_dataloader)

        # Perform evaluation over batches
        all_preds = []
        all_cams = []
        for batch_id, data in enumerate(process):
            with torch.no_grad():

                # Fetch batch
                if gpu_available:
                    data = Variable(data.float().cuda(output_device), requires_grad=False, volatile=True)
                else:
                    data = Variable(data.float(), requires_grad=False, volatile=True)

                # Perform inference
                output, feature = model(data)
                all_preds.append(output.data.cpu().numpy())
                
                # Class activation mapping (CAM) with EfficientGCN implementation (https://gitee.com/yfsong0709/EfficientGCNv1/blob/master/src/visualizer.py)
                if visualize:
                    weight = model.fcn.weight.squeeze().detach().cpu().numpy()
                    feature = feature[:,...].detach().cpu().numpy()
                    cam = np.einsum('kc,nctvm->nktvm', weight, feature)   
                    cam = cam[...,0]
                    all_cams.append(cam)

        # Concatenate
        preds = np.concatenate(all_preds)
        cams = np.concatenate(all_cams)
        
        # Compute softmax values
        softmax = np.zeros(preds.shape)
        for i in range(preds.shape[0]):
            softmax[i,:] = np.exp(preds[i,:]) / np.sum(np.exp(preds[i,:]), axis=0)
            
        preds_folds.append(softmax)
        cams_folds.append(cams)
    preds_folds = np.asarray(preds_folds)
    cams_folds = np.asarray(cams_folds)
        
    # Obtain and store risk of outcome, classification and uncertainty of classification
    if save:
        
        # Compute risk of outcome from ensemble of cross-validation folds
        ensemble_preds = np.median(preds_folds, axis=0)
        outcome_risk = np.median(ensemble_preds, axis=0)[1]
        
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
           
        # Store results in CSV
        csv_path = os.path.join(results_dir, '{0}_results.csv'.format(coords_path.split('/')[-1].split('orgcoords_')[1].split('.csv')[0]))
        csv_file = open(csv_path, 'w')
        headers = ['risk', 'prediction_threshold', 'positive_classification', 'certain_classification']
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()  
        results = {'risk': outcome_risk, 'prediction_threshold': prediction_threshold, 'positive_classification': outcome, 'certain_classification': certain}
        writer.writerow(results)
        csv_file.flush()
        csv_file.close()
    
    # Create and store CAM visualization
    if visualize:
        
        # Initialize visualization
        location = np.expand_dims(np.expand_dims(np.swapaxes(np.asarray(sample_coords), 0, 1), 1), -1)
        location[1,...] = 1.0 - location[1,...]
        location[0,...] *= 2000
        location[1,...] *= 2000
        
        # Combine positive and negative CAM
        combined_cams_body_parts_folds = []
        for crossval_fold in range(1, crossval_folds+1):
            cam = cams_folds[crossval_fold-1,...]
            cam_negative = cam[:,0,...]
            cam_negative_body_parts = np.median(cam_negative[:,:,:], axis=1)
            cam_positive = cam[:,1,...] 
            cam_positive_body_parts = np.median(cam_positive[:,:,:], axis=1)
            combined_cam_body_parts = cam_positive_body_parts - cam_negative_body_parts
            combined_cams_body_parts = []
            for n in range(combined_cam_body_parts.shape[0]):
                combined_cam_body_parts_sample = combined_cam_body_parts[n,...]
                combined_cam_body_parts_sample /= np.max(np.absolute(combined_cam_body_parts_sample))
                combined_cam_body_parts_sample = np.maximum(combined_cam_body_parts_sample, 0)
                combined_cams_body_parts.append(combined_cam_body_parts_sample)
            combined_cams_body_parts_folds.append(combined_cams_body_parts)
        combined_cams_body_parts_folds = np.asarray(combined_cams_body_parts_folds)
            
        # Aggregate combined CAM across cross-validation folds
        ensemble_combined_cams_body_parts = np.median(combined_cams_body_parts_folds, axis=0)
        accumulative_ensemble_combined_cams_body_parts = np.median(ensemble_combined_cams_body_parts, axis=0)
        accumulative_ensemble_combined_cams_body_parts = np.expand_dims(np.expand_dims(accumulative_ensemble_combined_cams_body_parts, 0), -1)
    
        # Store visualization
        plt.figure()
        plt.ion()
        plt.cla()
        plt.xlim(500, 2000)
        plt.ylim(500, 2000)
        plt.axis('off')
        x = location[0,0,:,0]
        y = location[1,0,:,0]
        connections = np.asarray(bone_conns) + 1
        c = []
        for v in range(len(body_parts)):
            r = accumulative_ensemble_combined_cams_body_parts[0,v,0]**(1/4)
            g = (1 - r)
            b = 0
            c.append([r, g, b])
            k = connections[v] - 1
            plt.plot([x[v], x[k]], [y[v], y[k]], '-o', c=np.array([0.1,0.1,0.1]), linewidth=1.0, markersize=0)
        plt.scatter(x, y, marker='o', c=c, s=100)
        plt.ioff()
        plt.savefig(os.path.join(results_dir, '{0}_cam.png'.format(coords_path.split('/')[-1].split('orgcoords_')[1].split('.csv')[0])), format='png')

        
if __name__ == '__main__':
    # Fetch arguments
    args = sys.argv[1:]
    coords_dir = args[0]
    files = os.listdir(coords_dir)
    coords_names = []
    for file in files:
        if '.' in file:
            file_format = file.split('.')[1].lower()
            if file_format in ['csv']:
                coords_names.append(file)
    for coords_name in tqdm(coords_names):
        predict(coords_dir, os.path.join(coords_dir, coords_name))
