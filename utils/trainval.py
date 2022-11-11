# Inspired by https://github.com/lshiwjx/2s-AGCN/blob/master/main.py

import copy
import json
import os
import pickle
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from art import tprint
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchprofile import profile_macs
from torchviz import make_dot
from tqdm import tqdm
from utils_functions import get_layer_metric_array, sum_arr
from zero_cost_proxies.grad_norm import calculate_grad_norm
from zero_cost_proxies.grasp import calculate_grasp
from zero_cost_proxies.synflow import calculate_synflow
from zero_cost_proxies.snip import calculate_snip

from utils import evaluate, feeder, graph


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    
class Operator():

    def __init__(self, processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold):
        # Initialize processed data dir
        self.processed_data_dir = processed_data_dir
        
        # Define fold of cross-validation experiment
        self.crossval_fold = crossval_fold
        
        # Initialize experiment
        self.crossval = True if self.crossval_fold is not None else False
        if self.crossval:
            self.experiment_dir = os.path.join(experiments_dir, 'crossval_{0}_val{1}'.format(candidate_num, self.crossval_fold))
        else:
            self.experiment_dir = os.path.join(experiments_dir, 'search_{0}'.format(candidate_num))
        os.makedirs(self.experiment_dir, exist_ok=True)
        # Initialize hyperparameters and candidate
        self.hyperparameters = hyperparameters
        self.candidate = candidate
        
        # Store candidate details as JSON file
        with open(os.path.join(self.experiment_dir, 'candidate.json'), 'w') as json_file:  
            json.dump(candidate, json_file)
        
        # Initialize tensorboardX
        self.train_writer = SummaryWriter(os.path.join(self.experiment_dir, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.experiment_dir, 'val'), 'val')
        
        # Initialize iteration number
        self.iteration = 0
        
        # Initialize validation results
        self.best_loss = 100000000000.0
        self.best_loss_epoch = None
        self.best_auc = 0.0
        self.best_auc_epoch = None
        
        # Initialize recent losses for median filter loss
        self.recent_losses = []
        
        # Initialize learning rate
        self.lr = self.hyperparameters['optimizer']['learning_rate']

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
        self.input_width = int(candidate['input_width'])
        self.num_input_modules = int(candidate['num_input_modules'])
        self.initial_block_type = candidate['initial_block_type']
        self.initial_residual = candidate['initial_residual']
        if candidate['input_temporal_scales'] == 'linear':
            self.input_temporal_scales = [i for i in range(1,int(self.num_input_modules)+1)]
        else:
            self.input_temporal_scales = [int(candidate['input_temporal_scales']) for i in range(self.num_input_modules)]
        self.initial_main_width = int(candidate['initial_main_width'])
        self.num_main_levels = int(candidate['num_main_levels'])
        self.num_main_level_modules = int(candidate['num_main_level_modules'])
        self.block_type = candidate['block_type']
        self.bottleneck_factor = int(candidate['bottleneck_factor'])
        self.residual = candidate['residual']
        if candidate['main_temporal_scales'] == 'linear':
            self.main_temporal_scales = [i for i in range(1,self.num_main_levels+1)]
        else:
            self.main_temporal_scales = [int(candidate['main_temporal_scales']) for i in range(self.num_main_levels)]
        self.temporal_kernel_size = int(candidate['temporal_kernel_size'])
        self.se_outer = False
        self.se_inner = False
        if candidate['se'] in ['inner', 'both']:
            self.se_inner = True
        if candidate['se'] in ['outer', 'both']:
            self.se_outer = True
        self.se_ratio = int(candidate['se_ratio'])
        self.relative_se = True if candidate['se_type'] == 'relative' else False
        self.swish_nonlinearity = True if candidate['nonlinearity'] == 'swish' else False
        self.attention = candidate['attention']
        self.spatial_pool = True if candidate['pool'] == 'spatial' else False
        self.init_model()
        
        # Initialize optimizer
        self.init_optimizer()
        
    def print_time(self):
        
        # Print current time
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        
        # Print log
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        
        if self.hyperparameters['optimizer']['print_log']:
            with open('{}/log.txt'.format(self.experiment_dir), 'a') as f:
                print(str, file=f)
    
    def record_time(self):
        
        # Record current time
        self.cur_time = time.time()
        
        return self.cur_time

    def split_time(self):
        
        # Compute time difference
        split_time = time.time() - self.cur_time
        self.record_time()
        
        return split_time

    def init_data(self, seed=1):
    
        # Intialize data dictionary
        self.data_loader = dict()

        # Initialize feeders
        if self.crossval:
            self.train_dataset = self.hyperparameters['crossval']['crossval_train_datasets'][self.crossval_fold - 1]
            self.val_dataset = self.hyperparameters['crossval']['crossval_val_datasets'][self.crossval_fold - 1]
        else:
            self.train_dataset = self.hyperparameters['search']['search_train_dataset']
            self.val_dataset = self.hyperparameters['search']['search_val_dataset']
        train_feeder = feeder.TrainFeeder(graph=self.graph_input, random_start=self.hyperparameters['augmentation']['random_start'], random_perturbation=self.hyperparameters['augmentation']['random_perturbation'], angle_candidate=[i for i in range(-self.hyperparameters['augmentation']['max_angle_candidate'], self.hyperparameters['augmentation']['max_angle_candidate']+1)], scale_candidate=[i/100 for i in range(int(self.hyperparameters['augmentation']['min_scale_candidate']*100), int(self.hyperparameters['augmentation']['max_scale_candidate']*100)+1)], translation_candidate=[i/100 for i in range(int(-self.hyperparameters['augmentation']['max_translation_candidate']*100), int(self.hyperparameters['augmentation']['max_translation_candidate']*100)+1)], roll_sequence=self.hyperparameters['augmentation']['roll_sequence'], standardize_rotation=self.hyperparameters['augmentation']['standardize_rotation'], input_temporal_resolution=self.hyperparameters['model']['input_temporal_resolution'], processed_data_dir=self.processed_data_dir, dataset=self.train_dataset, num_per_positive_sample=self.hyperparameters['balance']['train_num_per_positive_sample'], num_per_negative_sample=self.hyperparameters['balance']['train_num_per_negative_sample'], absolute=self.hyperparameters['features']['absolute'], relative=self.hyperparameters['features']['relative'], motion1=self.hyperparameters['features']['motion1'], motion2=self.hyperparameters['features']['motion2'], bone=self.hyperparameters['features']['bone'], bone_angle=self.hyperparameters['features']['bone_angle'])
        val_feeder = feeder.EvalFeeder(graph=self.graph_input, random_perturbation=False, standardize_rotation=self.hyperparameters['augmentation']['standardize_rotation'], input_temporal_resolution=self.hyperparameters['model']['input_temporal_resolution'], parts_distance=self.hyperparameters['evaluation']['parts_distance'], processed_data_dir=self.processed_data_dir, dataset=self.val_dataset, absolute=self.hyperparameters['features']['absolute'], relative=self.hyperparameters['features']['relative'], motion1=self.hyperparameters['features']['motion1'], motion2=self.hyperparameters['features']['motion2'], bone=self.hyperparameters['features']['bone'], bone_angle=self.hyperparameters['features']['bone_angle'])
        
        # Determine number of input channels
        self.input_channels = train_feeder.get_channels()

        # Training data
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=train_feeder,
            batch_size=self.hyperparameters['trainval']['trainval_batch_size'],
            shuffle=True,
            num_workers=self.hyperparameters['devices']['num_workers'],
            drop_last=True,
            worker_init_fn=init_seed(seed))

        # Validation data
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=val_feeder,
            batch_size=self.hyperparameters['trainval']['trainval_batch_size'],
            shuffle=False,
            num_workers=self.hyperparameters['devices']['num_workers'],
            drop_last=False,
            worker_init_fn=init_seed(seed))

    def init_model(self, is_double = False):
        
        # Set main processing unit
        self.output_device = int(self.hyperparameters['devices']['output_device'])
        # Initialize model
        model = __import__(self.hyperparameters['model']['model_script'], fromlist=['object'])
        self.model = model.Model(num_classes=2, graphs=[self.graph_input, self.graph_main] if len(self.labeling_mode) == 2 else [self.graph_input], input_channels=self.input_channels, edge_importance_weighting=self.hyperparameters['model']['edge_importance_weighting'], dropout=self.hyperparameters['model']['dropout'], num_input_branches=self.hyperparameters['model']['num_input_branches'], attention=self.attention, se_outer=self.se_outer, se_inner=self.se_inner, initial_residual=self.initial_residual, residual=self.residual, initial_block_type=self.initial_block_type, block_type=self.block_type, input_width=self.input_width, initial_main_width=self.initial_main_width, temporal_kernel_size=self.temporal_kernel_size, num_input_modules=self.num_input_modules, num_main_levels=self.num_main_levels, num_main_level_modules=self.num_main_level_modules, input_temporal_scales=self.input_temporal_scales, main_temporal_scales=self.main_temporal_scales, bottleneck_factor=self.bottleneck_factor, se_ratio=self.se_ratio, relative_se=self.relative_se, swish_nonlinearity=self.swish_nonlinearity, spatial_pool=self.spatial_pool)
        if is_double:
            self.model.double()
        if self.hyperparameters['devices']['gpu_available']:
            self.model = self.model.cuda(self.output_device)
        
        # Display model
        self.print_log(str(self.model))
        
        # Compute number of parameters
        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log("Number of parameters: {0}".format(self.num_parameters))
        
        # Compute number of floating point operations
        dummy_data = torch.from_numpy(np.zeros((2, self.input_channels, self.hyperparameters['model']['input_temporal_resolution'], self.hyperparameters['model']['input_spatial_resolution']))).float()
        if self.hyperparameters['devices']['gpu_available']:
            dummy_data = dummy_data.cuda(self.output_device)
        
        if is_double:
            dummy_data = dummy_data.double()

        macs = profile_macs(self.model, dummy_data) // 2
        self.num_flops = int(macs*2)
        self.print_log("Number of FLOPs per sample: {0}".format(self.num_flops))
        
        # Define loss function
        self.loss = nn.CrossEntropyLoss()
        if self.hyperparameters['devices']['gpu_available']:
            self.loss = self.loss.cuda(self.output_device)

    def init_optimizer(self):
        
        # Initialize Stochastic Gradient Descent optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hyperparameters['optimizer']['learning_rate'],
            momentum=self.hyperparameters['optimizer']['momentum'],
            nesterov=self.hyperparameters['optimizer']['nesterov'],
            weight_decay=self.hyperparameters['optimizer']['weight_decay'])
        
    def adjust_learning_rate(self, epoch, reduction_factor=0.1):
        
        # Adjust learning rate of epoch
        lr = self.hyperparameters['optimizer']['learning_rate'] * (reduction_factor ** np.sum(epoch >= np.array(self.hyperparameters['optimizer']['steps'])))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, epoch):
        
        # Initiate training mode
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.train_writer.add_scalar('epoch', epoch, self.iteration)
        
        # Adjust learning rate
        self.adjust_learning_rate(epoch, reduction_factor=self.hyperparameters['optimizer']['reduction_factor']) 
        
        # Fetch training data of epoch
        loader = self.data_loader['train']
        process = tqdm(loader, desc="Fetch training data for epoch")
                        
        # Perform iterations of training
        loss_iterations = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        for batch_id, (data, labels, video_ids, indices) in enumerate(process):
            
            # Fetch batch of data and labels
            if self.hyperparameters['devices']['gpu_available']:
                data = Variable(data.float().cuda(self.output_device), requires_grad=False) 
                labels = Variable(labels.long().cuda(self.output_device), requires_grad=False)
            else:
                data = Variable(data.float(), requires_grad=False) 
                labels = Variable(labels.long(), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # Forward pass
            output, feature = self.model(data)
            if batch_id == 0 and epoch == 0:
                self.train_writer.add_graph(self.model, data)
                make_dot(output, params=dict(list(self.model.named_parameters()))).render(os.path.join(self.experiment_dir, 'arch'), format="png")
                
            # Compute loss
            loss = self.loss(output, labels)

            # Bacward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_iterations.append(loss.data.item())
            timer['model'] += self.split_time()
            
            # Log batch statistics
            self.train_writer.add_scalar('loss', loss.data.item(), self.iteration)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.iteration)
            timer['statistics'] += self.split_time()
            
            self.iteration += 1
            
        # Aggregate loss across iterations
        loss_epoch = np.mean(loss_iterations)
        self.train_writer.add_scalar('loss_epoch', loss_epoch, epoch)
        
        # Log parameter histograms
        for name, param in self.model.named_parameters():
             self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        # Log time consumption of epoch and mean training loss
        self.print_log('\Training loss epoch: {:.5f}.'.format(loss_epoch))
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        
        return copy.deepcopy(self.model.state_dict())
    
    def val(self, epoch, save_preds=False):
    
        # Initiate evaluation mode
        self.model.eval()
        self.print_log('Validation epoch: {}'.format(epoch + 1))
        self.val_writer.add_scalar('epoch', epoch, self.iteration)
        
        # Fetch validation data
        process = tqdm(self.data_loader['val'], desc="Fetch validation data")

        # Perform validation over batches
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

        # Store predictions on validation set
        if save_preds:
            preds_object = []
            for video_id, pred, label in zip(video_ids, preds, labels):
                preds_object.append((video_id, pred, label))
            with open(os.path.join(self.experiment_dir, 'epoch-{0}_{1}_preds.pkl'.format(epoch + 1, self.val_dataset)), 'wb') as f:
                pickle.dump(preds_object, f)

        # Compute weighted mean loss
        loss_positives = []
        loss_negatives = []
        for sample_pred, sample_label in zip(preds, labels):
            if self.hyperparameters['devices']['gpu_available']:
                sample_loss = self.loss(torch.tensor(sample_pred).unsqueeze(0).cuda(self.output_device), torch.tensor(sample_label).unsqueeze(0).cuda(self.output_device)).cpu()
            else:
                sample_loss = self.loss(torch.tensor(sample_pred).unsqueeze(0), torch.tensor(sample_label).unsqueeze(0))
            if sample_label == 1:
                loss_positives.append(sample_loss)
            elif sample_label == 0:
                loss_negatives.append(sample_loss)
        val_positive_loss = np.mean(loss_positives)
        val_negative_loss = np.mean(loss_negatives)
        val_loss = (val_positive_loss + val_negative_loss)/2
        
        # Compute median filter loss
        self.recent_losses.append(val_loss)
        self.recent_losses = self.recent_losses[-self.hyperparameters['trainval']['loss_filter_size']:]
        filtered_loss = np.median(self.recent_losses)

        # Compute Area Under ROC Curve
        val_auc = evaluate.evaluate(preds, labels, video_ids, aggregate_binary=self.hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=self.hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=self.hyperparameters['evaluation']['median_aggregation'], prediction_threshold=self.hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=False, subject=True)
        val_window_auc = evaluate.evaluate(preds, labels, video_ids, aggregate_binary=self.hyperparameters['evaluation']['aggregate_binary'], aggregate_binary_threshold=self.hyperparameters['evaluation']['aggregate_binary_threshold'], median_aggregation=self.hyperparameters['evaluation']['median_aggregation'], prediction_threshold=self.hyperparameters['evaluation']['prediction_threshold'], threshold_metrics=False, subject=False)

        # Compare to best results
        
        # Weighted mean loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_loss_epoch = epoch + 1

        # Area Under ROC Curve
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_auc_epoch = epoch + 1

        # Log loss and validation metrics
        self.val_writer.add_scalar('mean_positive_loss', val_positive_loss, self.iteration)
        self.val_writer.add_scalar('mean_negative_loss', val_negative_loss, self.iteration)
        self.val_writer.add_scalar('loss_epoch', val_loss, epoch)
        self.val_writer.add_scalar('loss', val_loss, self.iteration)
        self.val_writer.add_scalar('filtered_loss', filtered_loss, self.iteration)
        self.val_writer.add_scalar('window_auc', val_window_auc, self.iteration)
        self.val_writer.add_scalar('subject_auc', val_auc, self.iteration)

        # Store results
        self.print_log('Validation results epoch {0}: Loss-->{1:.5f} Positive loss-->{2:.5f} Negative loss-->{3:.5f} Window Area Under ROC Curve-->{4:.5f} Subject Area Under ROC Curve-->{5:.5f}'.format(epoch + 1, val_loss, val_positive_loss, val_negative_loss, val_window_auc, val_auc))
            
        return val_loss, val_positive_loss, val_negative_loss, val_window_auc, val_auc
   
    def start(self):
        
        # Iterate over epochs
        num_epochs = self.hyperparameters['crossval']['crossval_num_epochs'] if self.crossval else self.hyperparameters['search']['search_num_epochs']
        for epoch in tqdm(range(num_epochs), desc="Epoch #: "):
            # Train
            model_state_dict = self.train(epoch)

            # Validate
            val_loss, val_positive_loss, val_negative_loss, val_window_auc, val_auc = self.val(epoch, save_preds=self.hyperparameters['crossval']['crossval_save_preds'] if self.crossval else self.hyperparameters['search']['search_save_preds'])

            # Verify critical epochs are met
            for critical_epoch, critical_value in zip(self.hyperparameters['crossval']['crossval_critical_epochs'] if self.crossval else self.hyperparameters['search']['search_critical_epochs'], self.hyperparameters['crossval']['crossval_critical_epoch_values'] if self.crossval else self.hyperparameters['search']['search_critical_epoch_values']):
                if epoch+1 >= critical_epoch:
                    if self.best_auc < critical_value:
                        return self.best_auc
                else:
                    break
            
            # Save model weights
            save_model = ((epoch + 1) % (self.hyperparameters['crossval']['crossval_save_interval'] if self.crossval else self.hyperparameters['search']['search_save_interval']) == 0)
            if save_model:
                weights_path = os.path.join(self.experiment_dir, 'epoch-{0}+loss-{1:.5f}+positive_loss-{2:.5f}+negative_loss-{3:.5f}+window_auc-{4:.5f}+subject_auc-{5:.5f}+iteration-{6}.pt'.format(epoch + 1, val_loss, val_positive_loss, val_negative_loss, val_window_auc, val_auc, int(self.iteration)))
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in model_state_dict.items()])
                torch.save(weights, weights_path)

        self.print_log('Best validation results: Loss-->{0:.5f}(epoch {1}) AUC-->{2:.5f}(epoch {3})'.format(self.best_loss, self.best_loss_epoch, self.best_auc, self.best_auc_epoch))

        return self.best_auc

    def get_zero_cost_score(self, method):
        if method == "grad_norm":
            score = calculate_grad_norm(self.model, self.data_loader, self.hyperparameters, self.output_device, self.loss)
        if method == "synflow":
            score = calculate_synflow(self.model, self.data_loader, self.hyperparameters, self.output_device, self.loss)
        if method == "snip":
            score = calculate_snip(self.model, self.data_loader, self.hyperparameters, self.output_device, self.loss)
        if method == "grasp":
            score = calculate_grasp(self.model, self.data_loader, self.hyperparameters, self.output_device, self.loss)

        return score
        
    def close(self):
        
        # Store candidate validation results as JSON file
        validation_results = {}
        validation_results['num_parameters'] = self.num_parameters
        validation_results['num_flops'] = self.num_flops
        validation_results['best_loss'] = self.best_loss
        validation_results['best_loss_epoch'] = self.best_loss_epoch
        validation_results['best_auc'] = self.best_auc
        validation_results['best_auc_epoch'] = self.best_auc_epoch
        with open(os.path.join(self.experiment_dir, 'validation_results.json'), 'w') as json_file:  
            json.dump(validation_results, json_file)
        
        # Close files
        self.train_writer.close()
        self.val_writer.close()
        
        # Clean memory
        del self.graph_input
        if len(self.labeling_mode) == 2:
            del self.graph_main
        del self.data_loader
        del self.model
        del self.optimizer
        if self.hyperparameters['devices']['gpu_available']:
            torch.cuda.empty_cache()
        return validation_results
        
        
def trainval(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold, train = False, zero_cost_method = None):
    # Initialize seeds
    init_seed(seed=hyperparameters['optimizer']['seed'])
    # Initialize operator
    operator = Operator(processed_data_dir, experiments_dir, candidate_num, candidate, hyperparameters, crossval_fold)
    # Run training and validation
    if zero_cost_method is not None:
        print(zero_cost_method)
        zc_score = operator.get_zero_cost_score(method=zero_cost_method)
    if train:
        auc = operator.start()

    # Close operator
    validation_results = operator.close()
    if train:
        return auc, validation_results
    else:
        return zc_score, validation_results


