# Inspired by https://github.com/lshiwjx/2s-AGCN/blob/master/main.py

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchprofile import profile_macs
from collections import OrderedDict
import numpy as np
import random
import json
from tqdm import tqdm
import pickle

from utils import graph, feeder, evaluate


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    
class Operator():

    def __init__(self, processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold):
        
        # Initialize processed data dir
        self.processed_data_dir = processed_data_dir
        
        # Define fold of cross-validation experiment
        self.crossval_fold = crossval_fold
        
        # Fetch experiment directory
        self.experiments_dir = experiments_dir
        self.experiment_dir = os.path.join(self.experiments_dir, 'crossval_{0}_val{1}'.format(candidate_num, self.crossval_fold))
        
        # Initialize hyperparameters and candidate
        self.hyperparameters = hyperparameters
        self.candidate = candidate

        # Initialize graph
        if candidate['graph'] == 'spatial':
            self.labeling_mode = ['spatial']
            self.disentangled_num_scales = [None]
        elif candidate['graph'].startswith('dis'):
            self.labeling_mode = ['disentangled']
            if candidate['graph'] == 'dis2':
                self.disentangled_num_scales = [2]
            elif candidate['graph'] == 'dis4':
                self.disentangled_num_scales = [4]
            elif candidate['graph'] == 'dis4+2':
                self.labeling_mode += ['disentangled']
                self.disentangled_num_scales = [4, 2]
        self.graph_input = graph.Graph(strategy=self.labeling_mode[0], body_parts=self.hyperparameters['graph']['body_parts'], neighbor_link=self.hyperparameters['graph']['neighbor_link'], center=self.hyperparameters['graph']['center'], bone_conns=self.hyperparameters['graph']['bone_conns'], thorax_index=self.hyperparameters['graph']['thorax_index'], pelvis_index=self.hyperparameters['graph']['pelvis_index'], disentangled_num_scales=self.disentangled_num_scales[0], use_mask=self.hyperparameters['graph']['use_mask'])
        if len(self.labeling_mode) == 2:
            self.graph_main = graph.Graph(strategy=self.labeling_mode[1], body_parts=self.hyperparameters['graph']['body_parts'], neighbor_link=self.hyperparameters['graph']['neighbor_link'], center=self.hyperparameters['graph']['center'], bone_conns=self.hyperparameters['graph']['bone_conns'], thorax_index=self.hyperparameters['graph']['thorax_index'], pelvis_index=self.hyperparameters['graph']['pelvis_index'], disentangled_num_scales=self.disentangled_num_scales[1], use_mask=self.hyperparameters['graph']['use_mask'])

        # Initialize data
        self.init_data(seed=self.hyperparameters['optimizer']['seed'])

        # Initialize model
        self.input_width = candidate['input_width']
        self.num_input_modules = candidate['num_input_modules']
        self.initial_block_type = candidate['initial_block_type']
        self.initial_residual = candidate['initial_residual']
        self.num_input_modules = candidate['num_input_modules']
        self.num_input_modules = candidate['num_input_modules']
        self.num_input_modules = candidate['num_input_modules']
        if candidate['input_temporal_scales'] == 'linear':
            self.input_temporal_scales = [i for i in range(1,self.num_input_modules+1)]
        else:
            self.input_temporal_scales = [int(candidate['input_temporal_scales']) for i in range(self.num_input_modules)]
        self.initial_main_width = candidate['initial_main_width']
        self.num_main_levels = candidate['num_main_levels']
        self.num_main_level_modules = candidate['num_main_level_modules']
        self.block_type = candidate['block_type']
        self.bottleneck_factor = candidate['bottleneck_factor']
        self.residual = candidate['residual']
        self.main_temporal_scales = candidate['bottleneck_factor']
        if candidate['main_temporal_scales'] == 'linear':
            self.main_temporal_scales = [i for i in range(1,self.num_main_levels+1)]
        else:
            self.main_temporal_scales = [int(candidate['main_temporal_scales']) for i in range(self.num_main_levels)]
        self.temporal_kernel_size = candidate['temporal_kernel_size']
        self.se_outer = False
        self.se_inner = False
        if candidate['se'] in ['inner', 'both']:
            self.se_inner = True
        if candidate['se'] in ['outer', 'both']:
            self.se_outer = True
        self.se_ratio = candidate['se_ratio']
        self.relative_se = True if candidate['se_type'] == 'relative' else False
        self.swish_nonlinearity = True if candidate['nonlinearity'] == 'swish' else False
        self.attention = candidate['attention']
        self.spatial_pool = True if candidate['pool'] == 'spatial' else False
        self.init_model()

    def init_data(self, seed=1):
    
        # Intialize data dictionary
        self.data_loader = dict()

        # Initialize feeder
        self.test_dataset = 'test'
        test_feeder = feeder.EvalFeeder(graph=self.graph_input, random_perturbation=False, standardize_rotation=self.hyperparameters['augmentation']['standardize_rotation'], input_temporal_resolution=self.hyperparameters['model']['input_temporal_resolution'], parts_distance=self.hyperparameters['evaluation']['parts_distance'], processed_data_dir=self.processed_data_dir, dataset=self.test_dataset, absolute=self.hyperparameters['features']['absolute'], relative=self.hyperparameters['features']['relative'], motion1=self.hyperparameters['features']['motion1'], motion2=self.hyperparameters['features']['motion2'], bone=self.hyperparameters['features']['bone'], bone_angle=self.hyperparameters['features']['bone_angle'])
        
        # Determine number of input channels
        self.input_channels = test_feeder.get_channels()
        
        # Test data
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_feeder,
            batch_size=self.hyperparameters['evaluation']['evaluation_batch_size'],
            shuffle=False,
            num_workers=self.hyperparameters['devices']['num_workers'],
            drop_last=False,
            worker_init_fn=init_seed(seed))

    def init_model(self):
        
        # Set main processing unit
        self.output_device = int(self.hyperparameters['devices']['output_device'])
        
        # Initialize model
        model = __import__(self.hyperparameters['model']['model_script'], fromlist=['object'])
        self.model = model.Model(num_classes=2, graphs=[self.graph_input, self.graph_main] if len(self.labeling_mode) == 2 else [self.graph_input], input_channels=self.input_channels, edge_importance_weighting=self.hyperparameters['model']['edge_importance_weighting'], dropout=self.hyperparameters['model']['dropout'], num_input_branches=self.hyperparameters['model']['num_input_branches'], attention=self.attention, se_outer=self.se_outer, se_inner=self.se_inner, initial_residual=self.initial_residual, residual=self.residual, initial_block_type=self.initial_block_type, block_type=self.block_type, input_width=self.input_width, initial_main_width=self.initial_main_width, temporal_kernel_size=self.temporal_kernel_size, num_input_modules=self.num_input_modules, num_main_levels=self.num_main_levels, num_main_level_modules=self.num_main_level_modules, input_temporal_scales=self.input_temporal_scales, main_temporal_scales=self.main_temporal_scales, bottleneck_factor=self.bottleneck_factor, se_ratio=self.se_ratio, relative_se=self.relative_se, swish_nonlinearity=self.swish_nonlinearity, spatial_pool=self.spatial_pool)
        if self.hyperparameters['devices']['gpu_available']:
            self.model = self.model.cuda(self.output_device)
        
        # Compute number of parameters
        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Compute number of floating point operations
        dummy_data = torch.from_numpy(np.zeros((2, self.input_channels, self.hyperparameters['model']['input_temporal_resolution'], self.hyperparameters['model']['input_spatial_resolution']))).float()
        if self.hyperparameters['devices']['gpu_available']:
            dummy_data = dummy_data.cuda(self.output_device)
        macs = profile_macs(self.model, dummy_data) // 2
        self.num_flops = int(macs*2)
        
        # Define loss function
        self.loss = nn.CrossEntropyLoss()
        if self.hyperparameters['devices']['gpu_available']:
            self.loss = self.loss.cuda(self.output_device)
        
        # Fetch epoch with highest performance
        with open(os.path.join(self.experiment_dir, 'validation_results.json'), 'r') as json_file: 
            self.best_epoch = json.load(json_file)['best_auc_epoch']
    
    def test(self, epoch, save_preds=False):
        
        # Load model weights
        for file in os.listdir(self.experiment_dir):
            if file.endswith('.pt') and file.startswith('epoch-{0}+'.format(self.best_epoch)):
                weights_path = os.path.join(
                    self.experiment_dir, file)
        weights = torch.load(weights_path)
        if self.hyperparameters['devices']['gpu_available']:
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)] for k, v in weights.items()])
        else:
            weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()]) 
        try:
            self.model.load_state_dict(weights)
        except:
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            state.update(weights)
            self.model.load_state_dict(state)
    
        # Initiate evaluation mode
        self.model.eval()
        
        # Fetch test data
        process = tqdm(self.data_loader['test'])

        # Perform evaluation over batches
        all_preds = []
        all_labels = []
        all_video_ids = []
        for batch_id, (data, labels, video_ids, indices) in enumerate(process):
            with torch.no_grad():

                # Fetch batch
                if self.hyperparameters['devices']['gpu_available']:
                    data = Variable(data.float().cuda(self.output_device), requires_grad=False, volatile=True)
                    labels = Variable(labels.long().cuda(self.output_device), requires_grad=False, volatile=True)
                else:
                    data = Variable(data.float(), requires_grad=False, volatile=True)
                    labels = Variable(labels.long(), requires_grad=False, volatile=True)

                # Perform inference
                output, feature = self.model(data)

                # Store predictions, labels and video ids
                all_preds.append(output.data.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_video_ids.append(video_ids)

        # Concatenate
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        video_ids = np.concatenate(all_video_ids)

        # Store predictions on test set
        if save_preds:
            preds_object = []
            for video_id, pred, label in zip(video_ids, preds, labels):
                preds_object.append((video_id, pred, label))
            with open(os.path.join(self.experiment_dir, '{0}_preds.pkl'.format(self.test_dataset)), 'wb') as f:
                pickle.dump(preds_object, f)

        # Compute weighted mean loss
        loss_positives = []
        loss_negatives = []
        for sample_pred, sample_label in zip(preds, labels):
            if self.hyperparameters['devices']['gpu_available']:
                sample_loss = self.loss(torch.tensor(sample_pred).unsqueeze(0).cuda(self.output_device), torch.tensor(sample_label).unsqueeze(0).cuda(self.output_device))
            else:
                sample_loss = self.loss(torch.tensor(sample_pred).unsqueeze(0), torch.tensor(sample_label).unsqueeze(0))
            if sample_label == 1:
                loss_positives.append(sample_loss)
            elif sample_label == 0:
                loss_negatives.append(sample_loss)
        test_positive_loss = float(np.mean(loss_positives))
        test_negative_loss = float(np.mean(loss_negatives))
        test_loss = (test_positive_loss + test_negative_loss)/2

        # Compute Area Under ROC Curve
        test_auc = evaluate.evaluate(preds, labels, video_ids, aggregate_binary=self.hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=self.hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=self.hyperparameters['evaluation']['median_aggregation'], prediction_threshold=self.hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=False, subject=True)
        test_window_auc = evaluate.evaluate(preds, labels, video_ids, aggregate_binary=self.hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=self.hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=self.hyperparameters['evaluation']['median_aggregation'], prediction_threshold=self.hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=False, subject=False)
            
        return preds, labels, video_ids, test_loss, test_positive_loss, test_negative_loss, test_window_auc, test_auc
   
    def start(self):
        
        # Test
        test_preds, test_labels, test_video_ids, self.test_loss, self.test_positive_loss, self.test_negative_loss, self.test_window_auc, self.test_auc = self.test(self.best_epoch, save_preds=self.hyperparameters['evaluation']['evaluation_save_preds'])
        
        return test_preds, test_labels, test_video_ids
        
    def close(self):
        
        # Store candidate test results as JSON file
        test_results = {}
        test_results['num_parameters'] = self.num_parameters
        test_results['num_flops'] = self.num_flops
        test_results['test_loss'] = self.test_loss
        test_results['test_positive_loss'] = self.test_positive_loss
        test_results['test_negative_loss'] = self.test_negative_loss
        test_results['test_window_auc'] = self.test_window_auc
        test_results['test_auc'] = self.test_auc
        test_results['test_epoch'] = self.best_epoch
        with open(os.path.join(self.experiment_dir, 'test_results.json'), 'w') as json_file:  
            json.dump(test_results, json_file)
        
        # Clean memory
        del self.graph_input
        if len(self.labeling_mode) == 2:
            del self.graph_main
        del self.data_loader
        del self.model
        if self.hyperparameters['devices']['gpu_available']:
            torch.cuda.empty_cache()
        
        
def test(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold):
    
    # Initialize seeds
    init_seed(seed=hyperparameters['optimizer']['seed'])
    
    # Initialize operator
    operator = Operator(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold)
    
    # Run testing
    preds, labels, video_ids = operator.start()
    
    # Close operator
    operator.close()
    
    return preds, labels, video_ids