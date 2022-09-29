# Inspired by https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py

import numpy as np
import torch
import torch.nn as nn

class Graph():
    """ The Graph to model the skeletons extracted by Markerless

    Args:
        body_parts (array): List of body parts in skeleton    
        neighbor_link (array): Neighboring body parts that are connected by bones in the skeleton
        center (int): Index of body part in center of skeleton
        bone_conns (array): Parent (i.e., body part closer to the center) of each body part 
        thorax_index (int): Index of thorax body part
        pelvis_index (int): Index of pelvis body part 
    
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in the ST-GCN paper  of Yan et al. (https://arxiv.org/abs/1801.07455).

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
        disentangled_num_scales (int): number of disentangled adjacency matrices, including hops distance from 0 to disentangled_num_scales-1
        use_mask (bool): If ``True``, adds a residual mask to the edges of the graph 
    """

    def __init__(self,
                 body_parts=['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'],
                 neighbor_link=[(0, 1), (2, 1), (3, 1), (1, 4), (9, 8), 
                              (10,9), (11,10), (5, 8), (6,5), (7,6),
                              (4,8), (12,8), (16,12), (17,16), (18,17),
                              (13,12), (14,13), (15,14)],
                 center=8,
                 bone_conns=[1,4,1,1,8,8,5,6,8,8,9,10,8,12,13,14,12,16,17],
                 thorax_index=8,
                 pelvis_index=12,
                 strategy='spatial',
                 max_hop=1,
                 dilation=1,
                 disentangled_num_scales=7,
                 use_mask=True):
        self.body_parts = body_parts
        self.neighbor_link = neighbor_link
        self.center = center
        self.bone_conns = np.array(bone_conns)
        self.thorax_index = thorax_index
        self.pelvis_index = pelvis_index
        
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self.disentangled_num_scales = disentangled_num_scales
        self.use_mask = use_mask

        self.get_edge()
        self.hop_dis = get_hop_distance(
            self.num_nodes, self.edge, max_hop=self.max_hop)
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_nodes = len(self.body_parts)
        self.self_link = [(i, i) for i in range(self.num_nodes)]
        self.edge = self.self_link + self.neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if self.strategy == 'uniform':
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        # https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
        elif self.strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        # https://github.com/kenziyuliu/MS-G3D/blob/master/graph/tools.py
        elif self.strategy == 'disentangled':
            outward = [(j, i) for (i, j) in self.neighbor_link]
            edges = self.neighbor_link + outward
            A_binary = get_adjacency_matrix(edges, self.num_nodes)
            A_binary_with_I = get_adjacency_matrix(edges + self.self_link, self.num_nodes)
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(self.disentangled_num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
            self.A = torch.Tensor(A_powers)
            if self.use_mask:
                # NOTE: the inclusion of residual mask appears to slow down training noticeably
                self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A.shape)), -1e-6, 1e-6)
                self.A = self.A + self.A_res
            self.A = self.A.view(self.disentangled_num_scales, self.num_nodes, self.num_nodes)
        else:
            raise ValueError("Do Not Exist This Strategy")

    # Used for finding the spine which can be used for rotating the body
    def get_thorax_pelvis_indices(self):        
        return self.thorax_index, self.pelvis_index


def get_hop_distance(num_nodes, edge, max_hop=1):
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_nodes, num_nodes)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_nodes = A.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)