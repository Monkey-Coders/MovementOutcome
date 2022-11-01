import os
import shutil
import random
import csv
import math
from tqdm import tqdm
import numpy as np

from utils.helpers import median_filter, coords_raw_to_norm


def copy_coords_files(individual_names, from_dir, to_dir):

    # Copy files to new directory
    os.makedirs(to_dir, exist_ok=True)
    for individual_name in tqdm(individual_names, desc="Copy files to new directory"):
        shutil.copyfile(os.path.join(from_dir, 'orgcoords_' + individual_name + '.csv'), os.path.join(to_dir, individual_name + '.csv'))

        
def generate_datasets(raw_dir, raw_coords_dir, raw_outcomes_file, test_size, crossval_folds):

    # Fetch individuals
    raw_file_names = os.listdir(raw_coords_dir)
    individual_names = []
    for raw_file_name in raw_file_names:
        if raw_file_name.lower().endswith('.csv'):
            individual_names.append(raw_file_name.split('.csv')[0].split('orgcoords_')[1])  
            
    # Divide individuals based on outcome
    positive_individuals = []
    negative_individuals = []
    with open(raw_outcomes_file, newline='') as outcomes_file:
        outcomes_reader = csv.reader(outcomes_file, delimiter=';', quotechar='|')
        header = next(outcomes_reader)
        for row in outcomes_reader:
            individual_name = row[0]
            if individual_name in individual_names:
                outcome = int(row[1])
                if outcome == 1 and individual_name not in positive_individuals:
                    positive_individuals.append(individual_name)
                elif outcome == 0 and individual_name not in negative_individuals:
                    negative_individuals.append(individual_name)
          
    # Shuffle individuals with positive and negative outcomes to create pseudo-random datasets
    random.seed(1433)
    random.shuffle(positive_individuals)
    random.shuffle(negative_individuals)
    
    # Determine crossval and test split
    crossval_test_split = 1.0 - test_size
    num_positive_individuals = len(positive_individuals)
    num_negative_individuals = len(negative_individuals)
    crossval_num_positive_individuals = int(num_positive_individuals * crossval_test_split)
    crossval_positive_individuals = positive_individuals[:crossval_num_positive_individuals]
    test_positive_individuals = positive_individuals[crossval_num_positive_individuals:]
    crossval_num_negative_individuals = int(num_negative_individuals * crossval_test_split)
    crossval_negative_individuals = negative_individuals[:crossval_num_negative_individuals]
    test_negative_individuals = negative_individuals[crossval_num_negative_individuals:]
    
    # Determine validation folds
    print("-- Generate cross-validation data")
    current_folds_remaining = crossval_folds
    for fold in range(1, crossval_folds+1):
        
        # Determine positive and negative individuals
        val_num_positive_individuals = math.ceil(len(crossval_positive_individuals) / current_folds_remaining)
        val_positive_individuals = crossval_positive_individuals[:val_num_positive_individuals]
        crossval_positive_individuals = crossval_positive_individuals[val_num_positive_individuals:]
        val_num_negative_individuals = math.ceil(len(crossval_negative_individuals) / current_folds_remaining)
        val_negative_individuals = crossval_negative_individuals[:val_num_negative_individuals]
        crossval_negative_individuals = crossval_negative_individuals[val_num_negative_individuals:]
        current_folds_remaining -= 1
        
        # Generate raw coords folders for test set
        print("--- Generate data of val{0}".format(fold))
        copy_coords_files(val_positive_individuals, raw_coords_dir, os.path.join(raw_dir, 'val{0}'.format(fold), 'positive'))
        copy_coords_files(val_negative_individuals, raw_coords_dir, os.path.join(raw_dir, 'val{0}'.format(fold), 'negative'))
    
    # Generate raw coords folders for test set
    print("-- Generate test data")
    copy_coords_files(test_positive_individuals, raw_coords_dir, os.path.join(raw_dir, 'test', 'positive'))
    copy_coords_files(test_negative_individuals, raw_coords_dir, os.path.join(raw_dir, 'test', 'negative'))
    
    
def perform_processing(raw_dir, processed_dir, crossval_folds, num_dimensions, num_joints, filter_stride):
    
    # Make skeleton sequences per dataset
    datasets = ['train{0}'.format(n) for n in range(1, crossval_folds+1)] + ['val{0}'.format(n) for n in range(1, crossval_folds+1)] + ['test']
    for dataset in datasets:
        
        # Fetch relevant directories
        if dataset.startswith('train'):
            fold = int(dataset.split('train')[1])
            raw_dataset_dirs = [os.path.join(raw_dir, 'val{0}'.format(n)) for n in range(1, fold)] + [os.path.join(raw_dir, 'val{0}'.format(n)) for n in range(fold+1, crossval_folds+1)]
        else:
            raw_dataset_dirs = [os.path.join(raw_dir, dataset)]
        
        # Obtain IDs and labels of individuals in dataset
        individual_dataset_outcomes = []
        individual_ids = []
        individual_labels = []
        for raw_dataset_dir in raw_dataset_dirs:
            positive_individual_file_names = os.listdir(os.path.join(raw_dataset_dir, 'positive'))
            for individual_file_name in positive_individual_file_names:
                if individual_file_name.endswith('.csv'):
                    individual_id = individual_file_name.split('.')[0] 
                    individual_dataset_outcomes.append((individual_id, raw_dataset_dir, 'positive'))
                    individual_ids.append(individual_id)
                    individual_labels.append(1)
            negative_individual_file_names = os.listdir(os.path.join(raw_dataset_dir, 'negative'))
            for individual_file_name in negative_individual_file_names:
                if individual_file_name.endswith('.csv'):
                    individual_id = individual_file_name.split('.')[0] 
                    individual_dataset_outcomes.append((individual_id, raw_dataset_dir, 'negative'))
                    individual_ids.append(individual_id)
                    individual_labels.append(0)
        num_individuals = len(individual_ids)
        
        # Generate processed skeleton sequences
        sequences = []
        for (individual_id, raw_dataset_dir, outcome) in tqdm(individual_dataset_outcomes):
            
            # Calculate median trunk length and median pelvis (assumes body keypoints of thorax and pelvis exist)
            trunk_lengths = []
            pelvis_xs = []
            pelvis_ys = []
            frames = {}
            body_parts = None
            with open(os.path.join(raw_dataset_dir, outcome, individual_id + '.csv'), newline='') as individual_file:
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
            individual_sequence = np.zeros((num_dimensions, num_frames, num_joints))
            for frame in frames.keys():

                # Apply median filter
                filter_frame_coords = median_filter(frames, frame, num_frames, filter_stride)

                # Centralize and normalize
                norm_frame_coords = coords_raw_to_norm(filter_frame_coords, median_pelvis_x, median_pelvis_y, median_trunk_length)
                individual_sequence[:, frame-1, :] = norm_frame_coords
            
            sequences.append(individual_sequence)
              
        # Determine maximum number of frames in skeleton sequence
        max_num_frames = 0
        for individual_sequence in sequences:
            _, num_frames, _ = individual_sequence.shape
            if num_frames > max_num_frames:
                max_num_frames = num_frames
      
        # Initialize data tensor
        data = np.zeros((num_individuals, num_dimensions, max_num_frames, num_joints), dtype=np.float32)

        # Feed data tensor with skeleton sequences
        for i, individual_sequence in enumerate(tqdm(sequences)):
            _, num_frames, _ = individual_sequence.shape
            data[i, :, 0:num_frames, :] = individual_sequence
            data[i, :, num_frames:, :] = None 
        
        # Store data tensor
        data_processed_path = '{0}/{1}_coords.npy'.format(processed_dir, dataset)
        np.save(data_processed_path, data)

        # Store tensor of sample IDs
        ids_processed_path = '{0}/{1}_ids.npy'.format(processed_dir, dataset)
        np.save(ids_processed_path, np.asarray(individual_ids))

        # Store label tensor
        labels_processed_path = '{0}/{1}_labels.npy'.format(processed_dir, dataset)
        np.save(labels_processed_path, np.asarray(individual_labels))
                        

def process(project_dir, processed_data_dir, test_size, crossval_folds, num_dimensions, num_joints, filter_stride):
    print('\n============================================================================================================================================\n')
    print('PROCESSING DATA\n')
    
    # Process raw coordinate files and outcomes
    processed_dir = processed_data_dir
    processed_exists = os.path.exists(processed_dir)
    if not processed_exists:
        
        # Make processed directory
        os.makedirs(processed_dir)
        
        # Split individuals into datasets (assumes there is no individual that is represented more than once)
        print('\n- Generating datasets')
        raw_dir = os.path.join(project_dir, 'data', 'raw')
        raw_coords_dir = os.path.join(raw_dir, 'coords')
        raw_outcomes_file = os.path.join(raw_dir, 'outcomes', 'outcomes.csv')
        generate_datasets(raw_dir, raw_coords_dir, raw_outcomes_file, test_size, crossval_folds)
        print('- Datasets generated') 
        
        # Create processed skeleton sequences with associated ids and labels
        print('\n- Generating skeleton sequences with ids and labels')
        perform_processing(raw_dir, processed_dir, crossval_folds, num_dimensions, num_joints, filter_stride) 
        print('- Skeleton sequences generated')
        print('\n============================================================================================================================================\n')