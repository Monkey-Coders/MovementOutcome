# Inspired by https://github.com/lshiwjx/2s-AGCN/blob/master/feeders/feeder.py

import numpy as np
from torch.utils.data import Dataset
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.helpers import random_start, random_perturbation, create_bone_motion_features, rotate, get_rotation_angle

class TrainFeeder(Dataset):
    def __init__(self, processed_data_dir, dataset, graph, 
                 num_per_positive_sample=5*5, num_per_negative_sample=5*1, input_temporal_resolution=150, random_start=False, 
                 random_perturbation=False, angle_candidate=[i for i in range(-45, 45+1)], 
                 scale_candidate=[i/100 for i in range(int(0.7*100), int(1.3*100)+1)], 
                 translation_candidate=[i/100 for i in range(int(-0.3*100), int(0.3*100)+1)], roll_sequence=False, 
                 standardize_rotation=True, use_mmap=True, absolute=True, relative=False, motion1=False, motion2=False, 
                 bone=False, bone_angle=False, debug=False):
        """ Feeder of training for skeleton-based action recognition with the In-Motion dataset
        Arguments:
            processed_data_dir: The parent directory of processed data
            dataset: The name of dataset for training
            graph: Graph to model skeletons
            num_per_positive_sample: Number of sequences from each positive sample
            num_per_negative_sample: Number of sequences from each negative sample
            input_temporal_resolution: The length of the output sequence
            random_start: If true, randomly choose a portion of the input sequence
            random_perturbation: If true, perform slight perturbations (rotation, scaling, and offsets) to the input sequence
            angle_candidate: Degree candidates for augmentation
            scale_candidate: Scaling candidates for augmentation
            translation_candidate: Translation candidates for augmentation
            roll_sequence: If true, repeat sequence until maximum T in dataset is reached
            standardize_rotation: If true, rotate sequences to start with a vertical body position
            use_mmap: If true, use mmap mode to load data, which can save the running memory
            absolute: If true, include absolute coordinates in input tensor
            relative: If true, include relative coordinates with respect to center joint in input tensor
            motion1: If true, include information about movement in one frame ahead of time in input tensor 
            motion2: If true, include information about movement in two frames ahead of time in input tensor 
            bone: If true, include bone vector in input tensor
            bone_angle: If true, include bone angles in input tensor
        """
        self.processed_data_dir = processed_data_dir
        self.dataset = dataset
        self.graph = graph
        self.num_per_positive_sample = num_per_positive_sample
        self.num_per_negative_sample = num_per_negative_sample
        self.input_temporal_resolution = input_temporal_resolution
        self.random_start = random_start
        self.random_perturbation = random_perturbation
        self.angle_candidate = angle_candidate
        self.scale_candidate = scale_candidate
        self.translation_candidate = translation_candidate
        self.roll_sequence = roll_sequence
        self.standardize_rotation = standardize_rotation
        self.use_mmap = use_mmap
        self.debug_slice = 100 if debug else None
        
        self.load_data()

        # Which features to use
        self.absolute = absolute
        self.relative = relative
        self.motion1 = motion1
        self.motion2 = motion2
        self.bone = bone
        self.bone_angle = bone_angle


    def load_data(self):

        # Load initial data structure
        if self.use_mmap:
            init_data = np.load(os.path.join(self.processed_data_dir, '{0}_coords.npy'.format(self.dataset)), mmap_mode='r')
        else:
            init_data = np.load(os.path.join(self.processed_data_dir, '{0}_coords.npy'.format(self.dataset)))
        
        # Load initial ID list
        init_ids = np.load(os.path.join(self.processed_data_dir, '{0}_ids.npy'.format(self.dataset)))

        # Load initial labels
        init_labels = np.load(os.path.join(self.processed_data_dir, '{0}_labels.npy'.format(self.dataset)))

        # Extract sample information
        samples = {}
        data = []
        ids = []
        labels = []
        sequence_parts = []
        for i, (sample_id, sample_label) in enumerate(zip(list(init_ids), list(init_labels))):
            samples[sample_id] = (init_data[i, ...], sample_label)


        # Balance dataset
        for sample_id in samples.keys():
            sample_data, sample_label = samples[sample_id]

            # Repeat sequence if specified
            if self.roll_sequence:
                C, dataset_T, V = sample_data.shape
                try:
                    sequence_T = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
                except:
                    sequence_T = dataset_T

                # Roll sequence
                num_repetitions = math.ceil(dataset_T / sequence_T)
                sequence_data = sample_data[:, :sequence_T, :]
                sample_data = np.zeros((C, num_repetitions*sequence_T, V))
                for n in range(num_repetitions):
                    sample_data[:, n*sequence_T:n*sequence_T+sequence_T ,:] = sequence_data
                sample_data = sample_data[:, :dataset_T, :]

            # Positive samples
            if sample_label == 1:
                for part in range(1, self.num_per_positive_sample+1):
                    data.append(sample_data)
                    ids.append(sample_id)
                    labels.append(sample_label)
                    sequence_parts.append((part, self.num_per_positive_sample))

            # Negative samples
            else:
                for part in range(1, self.num_per_negative_sample+1):
                    data.append(sample_data)
                    ids.append(sample_id)
                    labels.append(sample_label)
                    sequence_parts.append((part, self.num_per_negative_sample))

        self.data = np.array(data)[:self.debug_slice, :, :, :]
        self.ids = np.array(ids)[:self.debug_slice]
        self.label = np.array(labels)[:self.debug_slice]
        self.sequence_parts = sequence_parts[:self.debug_slice]
                    
    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self
    
    def get_channels(self):
        data, _, _, _ = self.__getitem__(0)
        in_channels, _, _ = data.shape
        return in_channels

    def __getitem__(self, index):
        
        # Fetch part information
        sample_data = np.array(self.data[index])
        sample_id = self.ids[index]
        sample_label = self.label[index]

        # Augment sample sequence part
        sequence_part, num_sequence_parts = self.sequence_parts[index]
        if self.random_start:
            sample_data = random_start(sample_data, self.input_temporal_resolution, roll_sequence=self.roll_sequence, sequence_part=sequence_part, num_sequence_parts=num_sequence_parts)
        else:
            sequence_length = sample_data.shape[1]
            if not self.roll_sequence:
                try:
                    sequence_length = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
                except:
                    pass
            part_size = math.floor(sequence_length / num_sequence_parts)
            window_start = (sequence_part - 1) * part_size if (sequence_part - 1) * part_size < (sequence_length - self.input_temporal_resolution) else sequence_length - self.input_temporal_resolution
            sample_data = sample_data[:, window_start:window_start+self.input_temporal_resolution, :]

        # Rotate the body to start the sequence with a vertical spine
        if self.standardize_rotation:
            angle = get_rotation_angle(sample_data, self.graph)
            sample_data = rotate(sample_data, angle)

        # Perform data-augmentation by scaling, rotating and transforming the sequence        
        if self.random_perturbation:
            sample_data = random_perturbation(sample_data, angle_candidate=self.angle_candidate, scale_candidate=self.scale_candidate, translation_candidate=self.translation_candidate)

        C, T, V = sample_data.shape
        sample_data = np.reshape(sample_data, (C, T, V, 1))
        sample_data = create_bone_motion_features(sample_data, self.graph.bone_conns)

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

        return sample_data, sample_label, sample_id, index
    
class EvalFeeder(Dataset):
    def __init__(self, processed_data_dir, dataset, graph, input_temporal_resolution=150, 
                 parts_distance=150, random_perturbation=False, angle_candidate=[i for i in range(-45, 45+1)], 
                 scale_candidate=[i/100 for i in range(int(0.7*100), int(1.3*100)+1)], 
                 translation_candidate=[i/100 for i in range(int(-0.3*100))], standardize_rotation=True,
                 use_mmap=True, absolute=True, relative=False, motion1=False, motion2=False, bone=False,
                 bone_angle=False, debug=False):
        """ Feeder of evaluation for skeleton-based action recognition with the In-Motion dataset
        Arguments:
            processed_data_dir: The parent directory of processed data
            dataset: The name of dataset for evaluation
            graph: Graph to model skeletons
            input_temporal_resolution: The length of the sequence parts
            parts_distance: The distance in timesteps between the start point of consecutive sequence parts 
            random_perturbation: If true, perform slight perturbations (rotation, scaling, and offsets) to the input sequence
            angle_candidate: Degree candidates for augmentation
            scale_candidate: Scaling candidates for augmentation
            translation_candidate: Translation candidates for augmentation
            standardize_rotation: If true, rotate sequences to start with a vertical body position
            use_mmap: If true, use mmap mode to load data, which can save the running memory
            absolute: If true, include absolute coordinates in input tensor
            relative: If true, include relative coordinates with respect to center joint in input tensor
            motion1: If true, include information about movement in one frame ahead of time in input tensor 
            motion2: If true, include information about movement in two frames ahead of time in input tensor 
            bone: If true, include bone vector in input tensor
            bone_angle: If true, include bone angles in input tensor
        """
        self.processed_data_dir = processed_data_dir
        self.dataset = dataset
        self.graph = graph
        self.input_temporal_resolution = input_temporal_resolution
        self.parts_distance = parts_distance
        self.random_perturbation = random_perturbation
        self.angle_candidate = angle_candidate
        self.scale_candidate = scale_candidate
        self.translation_candidate = translation_candidate
        self.standardize_rotation = standardize_rotation
        self.use_mmap = use_mmap
        self.debug_slice = 100 if debug else None

        self.load_data()

        # Which features to use
        self.absolute = absolute
        self.relative = relative
        self.motion1 = motion1
        self.motion2 = motion2
        self.bone = bone
        self.bone_angle = bone_angle

    def load_data(self):

        # Load initial data structure
        if self.use_mmap:
            init_data = np.load(os.path.join(self.processed_data_dir, '{0}_coords.npy'.format(self.dataset)), mmap_mode='r')
        else:
            init_data = np.load(os.path.join(self.processed_data_dir, '{0}_coords.npy'.format(self.dataset)))
        
        # Load initial ID list
        init_ids = np.load(os.path.join(self.processed_data_dir, '{0}_ids.npy'.format(self.dataset)))

        # Load initial labels
        init_labels = np.load(os.path.join(self.processed_data_dir, '{0}_labels.npy'.format(self.dataset)))

        # Extract sample information
        samples = {}
        for i, (sample_id, sample_label) in enumerate(zip(list(init_ids), list(init_labels))):
            samples[sample_id] = (init_data[i,...], sample_label)

        # Extract sequence parts
        data = []
        ids = []
        labels = []

        for sample_id in samples.keys():
            sample_data, sample_label = samples[sample_id]
            try:
                sample_num_frames = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
            except:
                sample_num_frames = sample_data.shape[1]

            # Construct sequence parts of self.input_temporal_resolution length and self.parts_distance distance to consecutive parts
            for start_frame in range(0, sample_num_frames, self.parts_distance):
                if start_frame > sample_num_frames - self.input_temporal_resolution:
                    data.append(sample_data[:, sample_num_frames - self.input_temporal_resolution:sample_num_frames, :])
                    ids.append(sample_id)
                    labels.append(sample_label)
                    break
                else:
                    data.append(sample_data[:, start_frame:start_frame+self.input_temporal_resolution, :])
                    ids.append(sample_id)
                    labels.append(sample_label)

        self.data = np.array(data)[:self.debug_slice, :, :, :]
        self.ids = np.array(ids)[:self.debug_slice]
        self.label = np.array(labels)[:self.debug_slice]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self
    
    def get_channels(self):
        data, _, _, _ = self.__getitem__(0)
        in_channels, _, _ = data.shape
        return in_channels

    def __getitem__(self, index):
        
        # Fetch part information
        sample_data = np.array(self.data[index])
        sample_id = self.ids[index]
        sample_label = self.label[index]

        # Augment sample sequence part
        # Rotate the body to start the sequence with a vertical spine
        if self.standardize_rotation:
            angle = get_rotation_angle(sample_data, self.graph)
            sample_data = rotate(sample_data, angle)

        # Perform data-augmentation by scaling, rotating and transforming the sequence
        if self.random_perturbation:
            sample_data = random_perturbation(sample_data, angle_candidate=self.angle_candidate, scale_candidate=self.scale_candidate, translation_candidate=self.translation_candidate)

        C, T, V = sample_data.shape
        sample_data = np.reshape(sample_data, (C, T, V, 1))
        sample_data = create_bone_motion_features(sample_data, self.graph.bone_conns)

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
        return sample_data, sample_label, sample_id, index