# _*_ coding:utf-8 _*_
#%%
import os
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from models.GCN import GCN
from models.N2RGNN import N2RGNN
from dataset import Dataset
from utils import accuracy


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="citeseer",
                    choices=['cora', 'citeseer','pubmed','dblp'], help='dataset')
parser.add_argument('--edge_rate', type=float, default=0.5,
                    help="noise of edges rate")                               # 添加噪声边的概率设为20%
parser.add_argument('--ptb_rate', type=float, default=0.5,
                    help="noise ptb_rate")                                    # 节点噪声概率设为20%
parser.add_argument('--epochs', type=int,  default=600,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='weight of loss of edge predictor')
parser.add_argument('--beta', type=float, default=1, 
                    help='weight of the loss on pseudo labels')
parser.add_argument('--t_small',type=float, default=0.1,
                    help='threshold of eliminating the edges')         
parser.add_argument('--p_u',type=float, default=0.8,        
                    help='threshold of adding pseudo labels')
parser.add_argument("--n_p", type=int, default=50, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')
parser.add_argument("--label_rate", type=float, default=0.05, 
                    help='rate of labeled data')                         
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                    help='type of noises')
parser.add_argument('--mlp_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--sigma', type=float, default=100,
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--c_num', type=float, default=0.00,
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--m_num', type=float, default=1.00,
                    help='the parameter to control the variance of sample weights in rec loss')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
np.random.seed(15) 

#%%
if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))    # 生成标签的索引
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]



if args.edge_rate>0:
    num_nodes = adj.shape[0]
    num_noise_edges = int((adj.nnz) * args.edge_rate)  
    graph = adj + adj.T    

    noise_edges = []
    while len(noise_edges) < num_noise_edges:
        node1, node2 = np.random.choice(num_nodes, size=2, replace=False)
        if graph[node1, node2] == 0:
            noise_edges.append((node1, node2))

    indices = np.array(noise_edges).T
    values = np.ones(num_noise_edges)
    noise_edges_sparse = sp.coo_matrix((values, indices), shape=adj.shape)

    noisy_adj_matrix = adj + noise_edges_sparse
else:
    noisy_adj_matrix = adj


from utils import noisify_with_P
ptb = args.ptb_rate         
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10, args.noise)
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y   
a_pure = labels[idx_train]     
a_label = labels



# %%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] =str(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic =True


esgnn = N2RGNN(args, device)
esgnn.fit(features, noisy_adj_matrix, noise_labels, idx_train, idx_val, a_label, args.c_num, args.m_num)  # 添加噪声
print("=====test set accuracy=======")
esgnn.test(idx_test)
print("===================================")


# %%
