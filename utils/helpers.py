import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pickle


def median_filter(frames, frame, num_frames, window_stride):
    
    # Ensure valid stride
    stride = window_stride if frame > window_stride else frame - 1
    stride = stride if frame + stride <= num_frames else num_frames - frame
    
    # Obtain window coords
    window_coords = np.zeros((frames[frame].shape[0], frames[frame].shape[1], 2 * stride + 1), dtype=frames[frame].dtype)
    for i, frame_index in enumerate(range(frame - stride, frame + stride + 1)):
        raw_coords = frames[frame_index]
        window_coords[:,:,i] = raw_coords
    filter_coords = np.median(window_coords, axis=2)
    
    return filter_coords


def coords_raw_to_norm(raw_coords, median_pelvis_x, median_pelvis_y, median_trunk_length, num_trunk_lengths=2):
        
    # Centralize with regards to median pelvis position
    centralized_coords = raw_coords
    centralized_coords[0,:] -= median_pelvis_x
    centralized_coords[1,:] -= median_pelvis_y

    # Normalize with regards to median trunk length
    norm_coords = centralized_coords / (2 * num_trunk_lengths * median_trunk_length)
    
    # Reverse y direction
    norm_coords[1,:] = -norm_coords[1,:]
    
    return norm_coords


def random_start(sample_data, window_size, roll_sequence=False, sequence_part=1, num_sequence_parts=1):
    # Inspired by https://github.com/lshiwjx/2s-AGCN/blob/master/feeders/tools.py
    
    # Fetch sequence information
    C, T, V = sample_data.shape
    if not roll_sequence:
        try:
            T = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
        except:
            pass
    
    # Select sequence starting point
    part_size = math.floor(T / num_sequence_parts)
    window_start_minimum = (sequence_part - 1) * part_size if (sequence_part - 1) * part_size < (T - window_size) else T - window_size
    window_start_maximum = sequence_part * part_size if (sequence_part * part_size) < (T - window_size) else T - window_size
    window_start = random.randint(window_start_minimum, window_start_maximum)
    
    return sample_data[:, window_start:window_start+window_size, :]


#  We calculate the vertical line (spine) of the body by finding the normal vector from the thorax onto the pelvis
#  This is used to find the degrees to rotate the body into a vertical position
def get_rotation_angle(sample_data, graph):
    thorax_index, pelvis_index = graph.get_thorax_pelvis_indices()
    thorax = sample_data[:, 0, thorax_index]
    pelvis = sample_data[:, 0, pelvis_index]

    spine = pelvis - thorax
    len_spine = np.sqrt(spine[0]**2 + spine[1]**2)
    rads_from_x_axis = np.arccos(spine[0]/len_spine) if len_spine > 0 else 0
    rads_to_rotate = -((np.pi / 2) + rads_from_x_axis) + np.pi
    angle_to_rotate = math.degrees(rads_to_rotate)
    return angle_to_rotate


def rotate(sample_data, angle):
    C, T, V = sample_data.shape
    rad = math.radians(-angle)
    a = np.full(shape=T, fill_value=rad)
    theta = np.array([[np.cos(a), -np.sin(a)],  # Rotation matrix
                      [np.sin(a), np.cos(a)]])  # xuanzhuan juzhen

    # Rotate joints for each frame
    for i_frame in range(T):
        xy = sample_data[0:2, i_frame, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        sample_data[0:2, i_frame, :] = new_xy.reshape(2, V)
    return sample_data


def scale(sample_data, scale_factor):
    sample_data[0:2, :, :] = sample_data[0:2, :, :] * scale_factor
    return sample_data


def translation(sample_data, delta_x, delta_y):
    sample_data[0, :, :] = sample_data[0, :, :] + delta_x
    sample_data[1, :, :] = sample_data[1, :, :] + delta_y
    return sample_data


def random_perturbation(sample_data,
                angle_candidate=[i for i in range(-45, 45+1)],
                scale_candidate=[i/100 for i in range(int(0.7*100), int(1.3*100)+1)],
                translation_candidate=[i/100 for i in range(int(-0.3*100), int(0.3*100)+1)]):

    sample_data = rotate(sample_data, random.choice(angle_candidate))
    sample_data = scale(sample_data, random.choice(scale_candidate))
    sample_data = translation(sample_data, random.choice(translation_candidate), random.choice(translation_candidate))
    return sample_data


# Create motion bone features like in PA-RES-GCN
# https://github.com/yfsong0709/ResGCNv1/tree/master/src/dataset/data_utils.py
def create_bone_motion_features(data, conn, center_joint=1):
    C, T, V, M = data.shape

    # Creates new features from the data tensor containing positions
    # Features: [[Absolute coordinates X, Absolute coordinates Y, Centered joint coordinates X, Centered joint coordinates X],
    #           [Motion stride 1 X, Motion stride 1 Y, Motion stride 2 X, Motion stride 2 Y],
    #           [Bone vector X, Bone vector Y, Bone angle X, Bone angle Y]]
    data_new = np.zeros((3, C*2, T, V, M))
    data_new[0,:C,:,:,:] = data

    # Center the joint coordinates with respect to the selected center joint
    for i in range(V):
        data_new[0,C:,:,i,:] = data[:,:,i,:] - data[:,:,center_joint,:]

    # Create motion from subsequent frames with stride 1 and 2
    for i in range(T-2):
        data_new[1,:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new[1,C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]

    # Calculate the bones connecting the joints and the bone-angles in vertical and horizontal direction
    for i in range(len(conn)):
        data_new[2,:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2,i,:,:,:], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2,C+i,:,:,:] = np.arccos(data_new[2,i,:,:,:] / bone_length)
    return data_new