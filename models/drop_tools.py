import numpy as np
import torch
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal

#删除边函数
def dropout_edges(adj, missing_link_rate):
    num_nodes = adj.shape[0]
    num_edges = adj.nnz  # 非零元素数量，即边的数量
    
    # 计算应该丢弃的边数
    num_edges_to_drop = int(num_edges * missing_link_rate)
    print("num to drop = ",num_edges_to_drop)
    
    if num_edges_to_drop > 0:
        # 获取边的索引
        row, col = adj.nonzero()
        # 随机选择要丢弃的边的索引
        drop_indices = np.random.choice(num_edges, size=num_edges_to_drop, replace=False)
        
        # 使用掩码过滤掉选择的边
        mask = np.ones(num_edges, dtype=bool)
        mask[drop_indices] = False
        
        # 删除mask个
        row = row[mask]
        col = col[mask]
        
        # 重新构建新的稀疏邻接矩阵
        dropped_adj_matrix = sp.coo_matrix((np.ones(row.size), (row, col)), shape=adj.shape)
        
        # 如果是无向图，还需要将反向边也丢弃
        #dropped_adj_matrix = dropped_adj_matrix + dropped_adj_matrix.T
        return dropped_adj_matrix
    else:
        return adj  # 没有丢弃任何边

#添加边噪声函数
def add_noise_edges_to_adj(origin_adj, adj, num_noise_edge, noise_rate):

    num_nodes = origin_adj.shape[0]
    num_edges = adj.nnz  # 原始边的数量

    # 计算噪声边的数量
    #num_noise_edges = int((origin_adj.nnz) * noise_rate)
    num_noise_edges = num_noise_edge
    
    # 构建无向图（原始邻接矩阵加上其转置）
    graph = adj + adj.T
    
    noise_edges = []
    
    # 随机选择噪声边，确保节点对之间没有原始边
    while len(noise_edges) < num_noise_edges:
        node1, node2 = np.random.choice(num_nodes, size=2, replace=False)
        if graph[node1, node2] == 0:  # 确保没有原有边
            noise_edges.append((node1, node2))

    # 构建噪声边的稀疏矩阵
    indices = np.array(noise_edges).T
    values = np.ones(len(noise_edges))  # 每条噪声边的值都为1
    noise_edges_sparse = sp.coo_matrix((values, indices), shape=adj.shape)

    # 将噪声边加入到原始邻接矩阵中
    noisy_adj_matrix = adj + noise_edges_sparse

    return noisy_adj_matrix



#添加 真边 函数
def add_real_edges_to_adj(adj, label, add_edge_rate):
    if add_edge_rate == 0.0:
        return adj
    
    num_nodes = adj.shape[0]
    num_edges = adj.nnz  # 原始边的数量

    # 计算添加边的数量
    num_real_edges = int(num_edges * add_edge_rate)
    
    # 构建无向图（原始邻接矩阵加上其转置）
    graph = adj + adj.T
    
    real_edges = []
    # 随机选择真边，确保节点对之间没有边，并且label要一致。
    while len(real_edges) < num_real_edges:
        node1, node2 = np.random.choice(num_nodes, size=2, replace=False)
        if graph[node1, node2] == 0 and label[node1] == label[node2]:  # 确保没有原有边，并且label要一致。
            real_edges.append((node1, node2))

    # 构建真实边的稀疏矩阵
    indices = np.array(real_edges).T
    values = np.ones(len(real_edges))  # 每条噪声边的值都为1
    real_edges_sparse = sp.coo_matrix((values, indices), shape=adj.shape)

    # 将噪声边加入到原始邻接矩阵中
    real_adj_matrix = adj + real_edges_sparse

    return real_adj_matrix




def add_flip_noise_gaussi(noisy_features, noise_rate, node_rate):
    
    num_nodes, num_features = noisy_features.shape
    
    # 随机选择要添加噪声的节点
    num_selected_nodes = int(node_rate * num_nodes)
    selected_nodes = np.random.choice(num_nodes, num_selected_nodes, replace=False)
    
    for node in selected_nodes:
        total_elements = num_features
        num_noisy = int(noise_rate * total_elements)
        # 随机选择要翻转的特征索引
        indices = np.random.choice(total_elements, num_noisy, replace=False)
        noise = np.random.normal(loc=0.0, scale=noise_rate, size=num_noisy)
        
        for i, idx in enumerate(indices):
            noisy_features[node, idx] += noise[i]
    return noisy_features    

# for循环运行速度很慢
# def add_flip_noise(noisy_features, noise_rate, node_rate):
#     # noisy_features = features.clone()
#     num_nodes, num_features = noisy_features.shape
    
#     # 随机选择要添加噪声的节点
#     num_selected_nodes = int(node_rate * num_nodes)
#     selected_nodes = np.random.choice(num_nodes, num_selected_nodes, replace=False)
    
#     for node in selected_nodes:
#         total_elements = num_features
#         num_noisy = int(noise_rate * total_elements)
#         # 随机选择要翻转的特征索引
#         indices = np.random.choice(total_elements, num_noisy, replace=False)
#         for idx in indices:
#             noisy_features[node, idx] = 1 - noisy_features[node, idx]
    
#     return noisy_features

# 局部翻转 适用cora、citesser
def add_flip_noise(noisy_features: sp.csr_matrix, noise_rate: float, node_rate: float) -> sp.csr_matrix:
    """
    对二值特征进行翻转噪声处理：
    随机选择 node_rate * N 个节点，对它们的所有特征，以 noise_rate 概率翻转 0↔1。
    """
    # dense = noisy_features.toarray()  # 转为密集矩阵
    if sp.issparse(noisy_features):
        dense = noisy_features.toarray().astype(np.int8)  # 或者 'bool'
    else:
        dense = noisy_features.astype(np.int8)
        
    num_nodes, num_features = dense.shape
    num_sel = int(node_rate * num_nodes)
    sel = np.random.choice(num_nodes, num_sel, replace=False)

    # mask = np.random.rand(num_sel, num_features) < noise_rate  # 生成 mask
    
    mask = (np.random.rand(num_sel, num_features) < noise_rate)
    # dense[sel] = dense[sel] ^ mask.astype(dense.dtype)         # 批量翻转
    dense[sel] = dense[sel] ^ mask.astype(np.int8)

    return sp.csr_matrix(dense)  # 转回稀疏矩阵



# 全局翻转 pubmed
def add_flip_noise_global(noisy_features: sp.csr_matrix, noise_rate: float) -> sp.csr_matrix:
    """
    对二值特征进行全局翻转噪声处理：
    对整个 feature 矩阵，以 noise_rate 的概率随机翻转 0 ↔ 1。
    """
    # 将稀疏矩阵转为密集矩阵，并转为整型
    dense = noisy_features.toarray().astype(np.int8)
    num_nodes, num_features = dense.shape

    # 全局生成 mask，与 dense 形状相同，每个位置以 noise_rate 概率为 True
    mask = (np.random.rand(num_nodes, num_features) < noise_rate)

    # 用 XOR 直接批量翻转：1^1=0, 0^1=1
    dense = dense ^ mask.astype(np.int8)

    # 返回稀疏格式
    return sp.csr_matrix(dense)

 

# 添加标签噪声
def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


def build_pair_p(size, noise):
    """
    生成配对噪声，有p概率变到最相似的类型标签中，计算混淆矩阵
    """
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    生成均匀噪声，原标签有p概率变到其他类型的标签，计算混淆矩阵
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    实操根据概率改变标签类型
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix 行随机矩阵（每一行元素之和为）
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)       # 生成随机数

    # 更新y标签
    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y