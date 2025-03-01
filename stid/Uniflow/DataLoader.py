import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
import os
from utils import tensor_to_adjacency_matrix
import utils
import networkx as nx
import pymetis
from tqdm import tqdm

class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        #print("min:", self._min, "max:", self._max, 'mean:{}, std:{}'.format(X.mean(), X.std()))

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

def edge_generate_grid(H,W):

    edges = []

    # Function to calculate the node index from (row, col)
    def node_index(row, col, width):
        return row * width + col

    # Generate edges for the grid
    for row in range(H):
        for col in range(W):
            current_node = node_index(row, col, W)
            # Connect to the right neighbor
            if col + 1 < W:
                right_node = node_index(row, col + 1, W)
                edges.append((current_node, right_node))
            # Connect to the bottom neighbor
            if row + 1 < H:
                bottom_node = node_index(row + 1, col, W)
                edges.append((current_node, bottom_node))

    return edges

def partition_graph(G, num_parts):
    (edgecuts, parts) = metis.part_graph(G, nparts=num_parts)
    return parts

def graph_split(num_nodes, adj, patch_size):
    '''
    ``adjacency[i]`` needs to be an iterable of vertices adjacent to vertex i.
      Both directions of an undirected graph edge are required to be stored.
    '''

    num = num_nodes // patch_size

    n_cuts, membership = pymetis.part_graph(num, adjacency=adj)

    node_split = []

    for i in range(max(membership)+1):
        node_split.append(torch.tensor(np.argwhere(np.array(membership) == i).ravel()))

    node_split = [i for i in node_split if len(i) > 0]

    lengths = [len(i) for i in node_split]
    print(set(lengths))

    return node_split

def data_load_index(args,dataset):
    
    batch_size = utils.select_batch_size(args, dataset)
    patch_size = utils.select_patch_size(args, dataset)

    folder_path = './dataset/train_data/UniFlow_dataset/{}.npy'.format(dataset) # T * H * W
    data = torch.tensor(np.load(folder_path)).float()

    new_test_batch = int(data.shape[0] * 0.2-args.pred_len-args.his_len)

    print('data:{}, shape:{}, PatchSize:{}, BatchSize:{}, NumBatch:{}'.format(dataset, data.shape, patch_size, batch_size, len(data)//batch_size))

    l, n, f = data.shape

    ts = np.load('./dataset/train_data/UniFlow_dataset/{}_ts.npy'.format(dataset))

    print(dataset)

    args.seq_len = args.his_len + args.pred_len

    if os.path.exists('./dataset/train_data/UniFlow_dataset/matrix_{}.json'.format(dataset)):
        f = open('./dataset/train_data/UniFlow_dataset/matrix_{}.json'.format(dataset),'r')
        matrix = json.load(f)
        edges = [k.split('_') for k in matrix['edges']]
        edges = [(int(edge[0]),int(edge[1])) for edge in edges]
        edges = torch.tensor(edges).long()
        nodes = matrix['nodes'].values()
        adj = [i for i in matrix['adj']]
        subgraphs = graph_split(len(nodes), adj, patch_size)

    else:
        edges = torch.tensor(edge_generate_grid(n//patch_size, f//patch_size)).long()
        nodes = [i for i in range(n*f//patch_size//patch_size)]
        subgraphs = None

    print('nodes:{}, edges:{}'.format(len(nodes), len(edges)))

    timestamps = torch.tensor(ts).long()

    my_scaler = MinMaxNormalization()
    MAX = torch.max(data).item()
    MIN = torch.min(data).item()
    
    my_scaler.fit(np.array([MIN, MAX]))

    if args.data_norm == 1:
        data = my_scaler.transform(data.reshape(-1,1)).reshape(data.shape)

    args.mean_value = torch.mean(data).item()

    num_samples = l - (args.his_len + args.pred_len) + 1
    train_num = round(num_samples * 0.6)
    valid_num = round(num_samples * 0.2)
    test_num = num_samples - train_num - valid_num

    index_list = []
    for t in range(args.his_len, num_samples + args.his_len):
        index = (t-args.his_len, t, t+args.pred_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    train_index = torch.tensor(train_index)
    test_index = torch.tensor(test_index)
    valid_index = torch.tensor(valid_index)
    
    if args.few_ratio < 1 and (dataset in args.few_data or args.few_data in dataset):
        train_index  = train_index[:int(len(train_index )*args.few_ratio)]

    train_index = th.utils.data.DataLoader(train_index, num_workers=4, batch_size=batch_size, shuffle=True) 
    test_index = th.utils.data.DataLoader(test_index, num_workers=4, batch_size =  batch_size, shuffle=False)
    val_index = th.utils.data.DataLoader(valid_index, num_workers=4, batch_size = batch_size, shuffle=False)

    return data, timestamps, train_index, test_index, val_index, my_scaler, edges, subgraphs


def data_load_index_mix(args):
    data_all = {}
    train_index_all = []
    test_index_all = []
    val_index_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('*'):
        data, ts, train_index, test_index, val_index, my_scaler, matrix, subgraphs = data_load_index(args,dataset_name)
        data_all[dataset_name] = [data, ts]
        if dataset_name not in args.few_data or args.few_ratio > 0: 
            train_index_all.append([dataset_name, train_index, matrix, subgraphs])
        test_index_all.append([test_index, matrix, subgraphs])
        val_index_all.append([val_index, matrix, subgraphs])
        my_scaler_all[dataset_name] = my_scaler

    train_index_all = [(n, j, m, s) for n, index, m, s in train_index_all for j in index]
    random.seed(1111)
    random.shuffle(train_index_all)

    return data_all, train_index_all, test_index_all, val_index_all, my_scaler_all



def data_load_index_main(args):
    data, train_index, test_index, val_index, scaler = data_load_index_mix(args)

    return data, train_index, test_index, val_index, scaler

