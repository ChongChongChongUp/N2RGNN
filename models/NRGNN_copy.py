# _*_ coding:utf-8 _*_
#%%
import copy
import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.GCN import GCN
from utils import accuracy, sparse_mx_to_torch_sparse_tensor, count_degree, tensor_to_sparse
from torch.utils.tensorboard import SummaryWriter


class NRGNN:
    def __init__(self, args, device):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None

    def fit(self, features, adj, labels, idx_train, idx_val, pure, count_num, m_num):

        args = self.args
        global pure_label
        pure_label = pure
        # 邻接矩阵->边索引
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        # 特征->张量
        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)
        pure_label = torch.LongTensor(np.array(pure_label)).to(self.device)

        # 标签->张量
        self.edge_index = edge_index
        self.features = features
        # self.labels = labels
        self.labels = copy.deepcopy(labels)
        self.ori_labels = labels
        self.ntime = np.zeros(len(labels))
        self.nlabel = np.zeros(len(labels))
        am = torch.LongTensor(np.array(self.nlabel)).to(self.device)
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)
        self.idx_train = copy.deepcopy(idx_train)       # 自己加上去的
        self.solid_train = idx_train
        self.count_num = count_num
        self.m_num = m_num
        # 创建GCN分类器
        # 创建伪标签预测器
        self.predictor = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)


        self.model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        # 原本的边预测器
        # self.estimator = EstimateAdj(features.shape[1], args, idx_train ,device=self.device).to(self.device)

        # 使用RSGNN中的方法创建边预测器
        self.edge_predictor = PreEdge(edge_index, features, args, device=self.device).to(self.device)
        # obtain the condidate edges linking unlabeled and labeled nodes
        # 获取候选边索引
        # self.pred_edge_index = self.get_train_edge(edge_index,features, self.args.n_p, idx_train)
        self.pred_edge_index = self.edge_predictor.poten_edge_index


        # 原始优化器
        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.estimator.parameters())+ list(self.predictor.parameters()),
        #                        lr=args.lr, weight_decay=args.weight_decay)
        # 新的优化器
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.edge_predictor.parameters()) + list(self.predictor.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
        self.loop_optimizer = optim.Adam(list(self.edge_predictor.parameters()) + list(self.predictor.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
        # Train model
        t_total = time.time()
        writer = SummaryWriter('log_train_try01')

        early_stopping = EarlyStopping(patience=500)
        for epoch in range(args.epochs):
            # self.train(epoch, features, edge_index, idx_train, idx_val, writer)
            for k in range(2):
                # self.train_loop(epoch, features, self.pred_edge_index, self.idx_train, idx_val, writer, early_stopping)
                self.train_loop(epoch, features, edge_index, self.idx_train, idx_val, writer, early_stopping)
            self.train(epoch, features, edge_index, self.idx_train, idx_val, writer, early_stopping)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        amax_indices = torch.argmax(self.best_pred, dim=1)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weights)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train_loop(self, epoch, features, edge_index, idx_train, idx_val, writer,early_stopping):
        args = self.args
        t = time.time()
        self.predictor.train()      # 伪标签预测
        self.loop_optimizer.zero_grad()
        # 获取表示和边预测损失(原始方法)
        representations, rec_loss = self.edge_predictor(edge_index, features)        # 新方法
        degrees_indices, degrees_nums = count_degree(tensor_to_sparse(self.pred_edge_index, features.shape[0]))  # 返回前n%的度对应索引和度数
        degrees_indices = torch.tensor(degrees_indices).to(self.device)
        degrees_nums = torch.tensor(degrees_nums).to(self.device)
        noise_index = torch.tensor(np.setdiff1d(self.solid_train, self.idx_train)).to(self.device)      # 噪声节点
        # 原始伪标签挖掘的预测结果，获取权重
        predictor_weights = self.edge_predictor.get_estimated_weights(representations)
        # 修改权重部分
        end_points = self.pred_edge_index[1]
        degrees_map = torch.zeros_like(predictor_weights, dtype=torch.int32).to(self.device)
        masked = torch.eq(end_points[:, None], degrees_indices)
        matching_indices = torch.nonzero(masked, as_tuple=False)
        degrees_nums = degrees_nums.type(torch.int32)
        degrees_map[matching_indices[:, 0]] = degrees_nums[matching_indices[:, 1]]
        predictor_weights_tmp = predictor_weights.clone()
        #predictor_weights *= torch.pow(self.m_num, self.count_num*degrees_map)
        # 伪标签挖掘，通过GCN网络后得到每个类别的概率
        log_pred = self.predictor(features, self.pred_edge_index, predictor_weights)
        self.nlog = log_pred
        # 获取伪标签和新候选边
        if self.best_pred == None:
            pred = F.softmax(log_pred,dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)             # 添加伪标签
        else:
            # 保存最佳伪标签
            pred = self.best_pred
        model_edge_index = self.pred_edge_index
        estimated_weights = predictor_weights
        pred_max, _ = torch.max(self.best_pred, dim=1)  # 预测最大的概率
        predlabel = torch.argmax(self.best_pred, dim=1)
        trust_idx = []
        noise_idx = []
        if epoch >= 300:
            for i in idx_train:
                if i not in trust_idx:
                    nei_node = self.pred_edge_index[1, self.pred_edge_index[0] == i]  # 节点i的邻居节点索引
                    nei_label = self.labels[nei_node]
                    max_label, _ = torch.mode(nei_label)  # 取邻居节点中出现最多的种类
                    loss_rl = F.cross_entropy(self.nlog[i].unsqueeze(0), self.labels[i].unsqueeze(0)).item()
                    if loss_rl < 0.30 and predlabel[i] == max_label:
                        trust_idx.append(i)
                        self.labels[i] = max_label
                    elif predlabel[i] == max_label:
                        self.labels[i] = max_label
                    else:
                        noise_idx.append(i)
          
        self.idx_train = np.setdiff1d(self.idx_train, noise_idx)
        idx_train = copy.deepcopy(self.idx_train)
        idx_add_cpu = self.idx_add.cpu()
        ##修改
        weight_label_loss = label_loss(self.pred_edge_index, predictor_weights_tmp, self.labels,
                                        np.concatenate((self.idx_train,idx_add_cpu)), self.device)
        ##
        # 定义伪标签挖掘损失
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])

        # 总损失函数(rec_loss是边预测器损失）
        total_loss = loss_pred + self.args.alpha * rec_loss  + 10 * weight_label_loss
        if epoch>=100:
            early_stopping(total_loss)
        total_loss.backward()
        self.loop_optimizer.step()

        # 计算训练集准确率
        self.predictor.eval()
        pred = F.softmax(self.predictor(features, self.pred_edge_index, predictor_weights),dim=1)
        acc_pred_val = accuracy(pred[idx_val], self.ori_labels[idx_val])
        # 保存效果最好的模型参数
        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weights = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
        self.pred_edge_index = self.edge_predictor.poten_edge_index 

            


    def train(self, epoch, features, edge_index, idx_train, idx_val, writer,early_stopping):
        args = self.args

        t = time.time()
        self.model.train()
        self.predictor.train()      # 伪标签预测
        self.optimizer.zero_grad()

        # obtain representations and rec loss of the estimator
        # 获取表示和边预测损失(原始方法)
        # representations, rec_loss = self.estimator(edge_index,features)
        
        representations, rec_loss = self.edge_predictor(edge_index, features)        # 新方法
        

        degrees_indices, degrees_nums = count_degree(tensor_to_sparse(self.pred_edge_index, features.shape[0]))  # 返回前n%的度对应索引和度数
        degrees_indices = torch.tensor(degrees_indices).to(self.device)
        degrees_nums = torch.tensor(degrees_nums).to(self.device)

        noise_index = torch.tensor(np.setdiff1d(self.solid_train, self.idx_train)).to(self.device)      # 噪声节点

        # prediction of accurate pseudo label miner
        # 原始伪标签挖掘的预测结果，获取权重
        predictor_weights = self.edge_predictor.get_estimated_weights(representations)
        # 修改权重部分
        end_points = self.pred_edge_index[1]
        # mask = (end_points.unsqueeze(1) == noise_index.unsqueeze(0)).any(dim=1)
        # predictor_weights[mask] *= 0.9

        
        degrees_map = torch.zeros_like(predictor_weights, dtype=torch.int32).to(self.device)
        masked = torch.eq(end_points[:, None], degrees_indices)
        matching_indices = torch.nonzero(masked, as_tuple=False)
        degrees_nums = degrees_nums.type(torch.int32)
        degrees_map[matching_indices[:, 0]] = degrees_nums[matching_indices[:, 1]]

        predictor_weights_tmp = predictor_weights.clone()
        #predictor_weights *= torch.pow(self.m_num, self.count_num*degrees_map)
        # predictor_weights *= torch.pow(0.90, 0.01*degrees_map)


        

        # 伪标签挖掘，通过GCN网络后得到每个类别的概率
        log_pred = self.predictor(features, self.pred_edge_index, predictor_weights)

        self.nlog = log_pred

        # obtain accurate pseudo labels and new candidate edges
        # 获取伪标签和新候选边
        if self.best_pred == None:
            
            pred = F.softmax(log_pred,dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)             # 添加伪标签
        else:
            # 保存最佳伪标签
            pred = self.best_pred

        # 更改标签
        # if epoch >= 150:
        #     self.labels[idx_train] = torch.argmax(self.best_pred, dim=1)[idx_train]




        # prediction of the GCN classifier
        # GCN分类器预测结果(原始
        # 这一部分对应的是原来的Assign Edges，需要在这里加入一个判别，判断标签节点的可靠性
        # idx_train = self.get_relia_labels(self.pred_edge_index, features, idx_train)
        # estimated_weights = self.estimator.get_estimated_weights(self.unlabel_edge_index,representations)
        # estimated_weights = self.edge_predictor.get_weights(self.unlabel_edge_index, representations)
        #
        # estimated_weights = torch.cat([predictor_weights, estimated_weights],dim=0)
        # model_edge_index = torch.cat([self.pred_edge_index,self.unlabel_edge_index],dim=1)
        model_edge_index = self.pred_edge_index
        estimated_weights = predictor_weights
        output = self.model(features, model_edge_index, estimated_weights)
        pred_model = F.softmax(output, dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1-eps)

        pred_max, _ = torch.max(self.best_pred, dim=1)  # 预测最大的概率
        predlabel = torch.argmax(self.best_pred, dim=1)
        trust_idx = []
        noise_idx = []
        if epoch >= 300:
            for i in idx_train:
                if i not in trust_idx:
                    nei_node = self.pred_edge_index[1, self.pred_edge_index[0] == i]  # 节点i的邻居节点索引
                    nei_label = self.labels[nei_node]
                    max_label, _ = torch.mode(nei_label)  # 取邻居节点中出现最多的种类
                    loss_rl = F.cross_entropy(self.nlog[i].unsqueeze(0), self.labels[i].unsqueeze(0)).item()
                    if loss_rl < 0.30 and predlabel[i] == max_label:
                        trust_idx.append(i)
                        self.labels[i] = max_label
                    elif predlabel[i] == max_label:
                        self.labels[i] = max_label
                    else:
                        noise_idx.append(i)

        # for j in self.idx_unlabel:
        #     if pred_max[j] >= 0.5:       # 最初始的版本就是修改idx_train>0.3的情况
        #         self.labels[j] = predlabel[j]
        # # 迭代时候增加的
        # self.edge_predictor = PreEdge(self.pred_edge_index, features, args, device=self.device).to(self.device)
        # self.pred_edge_index = self.edge_predictor.poten_edge_index
        # representations, rec_loss = self.edge_predictor(self.pred_edge_index, features)
        # predictor_weights = self.edge_predictor.get_estimated_weights(representations)
        #
        # log_pred = self.predictor(features, pred_edge_index, predictor_weights)

        self.idx_train = np.setdiff1d(self.idx_train, noise_idx)
        idx_train = copy.deepcopy(self.idx_train)

        ##修改
        weight_label_loss = label_loss(self.pred_edge_index, predictor_weights_tmp, self.labels,
                                       self.idx_train, self.device)
        ##np.concatenate((self.idx_train,self.idx_add))

        # loss from pseudo labels
        # 定义伪标签损失
        loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
        # loss of accurate pseudo label miner
        # 定义伪标签挖掘损失
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])


        # loss of GCN classifier
        # 定义GCN分类损失
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])
        # 总损失函数(rec_loss是边预测器损失）
        total_loss = loss_gcn + loss_pred + self.args.alpha * rec_loss  +self.args.beta * loss_add+\
            10*weight_label_loss
        #self.args.beta * loss_add+

        if epoch>=100:
            early_stopping(total_loss)
        total_loss.backward()
        self.optimizer.step()

        # 计算训练集准确率
        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])
        # print("=====Train Accuray=====")
        # print(f"Epoch {epoch}: acc:{acc_train:.4f}")
        # writer.add_scalar('without relia', acc_train.item(), epoch)
        # Evaluate validation set performance separately,
        # 在验证集上评估GCN和边预测器的表现
        self.model.eval()
        self.predictor.eval()
        pred = F.softmax(self.predictor(features, self.pred_edge_index, predictor_weights),dim=1)


        output = self.model(features, model_edge_index, estimated_weights.detach())
        acc_pred_val = accuracy(pred[idx_val], self.ori_labels[idx_val])
        acc_val = accuracy(output[idx_val], self.ori_labels[idx_val])
        # 保存效果最好的模型参数
        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weights = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
            



        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))



        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_pred: {:.4f}'.format(loss_pred.item()),
                      'loss_add: {:.4f}'.format(loss_add.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'acc_pred_val: {:.4f}'.format(acc_pred_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
                print('Size of add idx is {}'.format(len(self.idx_add)))
        # print("=====Validation Accuray=====")
        # print(f'Epoch {epoch}: acc:{acc_pred_val.item():.4f}')

        self.pred_edge_index = self.edge_predictor.poten_edge_index

    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        # labels = self.labels
        labels = self.ori_labels

        self.predictor.eval()
        estimated_weights = self.best_pred_graph        # 边预测器的权重
        # pred_edge_index = torch.cat([self.edge_index,self.pred_edge_index],dim=1)
        output = self.predictor(features, self.pred_edge_index, estimated_weights)      # 伪标签预测
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tPredictor results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        self.model.eval()
        estimated_weights = self.best_graph             # 边预测器的权重
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index, estimated_weights)               # GCN分类预测
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tGCN classifier results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)


    def get_model_edge(self, pred):     # 添加伪标签
        
        # 获取预测标签大于阈值的未标记节点的索引，对应公式10
        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u)].cpu()
        # 构建边的索引，其中未标记节点与预测为正类的节点之间存在边
        row = self.idx_unlabel.repeat(len(idx_add)).cpu()
        col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
        mask = (row!=col)
        unlabel_edge_index = torch.stack([row[mask],col[mask]], dim=0)

        return unlabel_edge_index, idx_add      # 返回添加的边的索引和伪标签节点的索引，unlabel_edge_index连接了未标记节点和伪标签节点

    def get_relia_labels(self, edge_index, features, idx_train):    # 判别标签节点和邻居节点的相似度
        for index in idx_train:
            indices = edge_index[1, edge_index[0] == index]
            sim = torch.div(torch.matmul(features[index], features[indices].T),
                            features[index].norm() * features[indices].norm(dim=1))
            aaab = sim.sum()
            if aaab <= 5:
                idx_train = idx_train[idx_train != index]
        return idx_train


# %%
class PreEdge(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    重建图
    """

    def __init__(self, edge_index, features, args, device='cuda'):
        super(PreEdge, self).__init__()
        self.etm_edge = nn.Sequential(nn.Linear(features.shape[1], args.mlp_hidden),
                                      nn.ReLU(),
                                      nn.Linear(args.mlp_hidden, args.mlp_hidden))
        self.device = device
        self.args = args
        self.poten_edge_index = self.get_poten_edge(edge_index, features, args.n_p)
        self.features_diff = torch.cdist(features, features, 2)  # 计算features样本之间的的欧氏距离
        self.estimated_weights = None

    def get_poten_edge(self, edge_index, features, n_p):
        # 保留候选边，对应NRGNN中的get_train_edge函数
        if n_p == 0:
            return edge_index

        poten_edges = []
        for i in range(len(features)):
            # 计算相似度
            sim = torch.div(torch.matmul(features[i], features.T), features[i].norm() * features.norm(dim=1))
            # 选择100个正样本
            _, indices = sim.topk(n_p)
            poten_edges.append([i, i])
            indices = set(indices.cpu().numpy())
            # 将于节点i相邻的节点添加到indices中
            indices.update(edge_index[1, edge_index[0] == i])
            for j in indices:
                if j > i:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        return poten_edges

    def forward(self, edge_index, features):

        representations = self.etm_edge(features)
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations, rec_loss

    # def get_estimated_weights(self, representations):
    #     x0 = representations[self.poten_edge_index[0]]
    #     x1 = representations[self.poten_edge_index[1]]
    #     output = torch.sum(torch.mul(x0, x1), dim=1)
    #     # 此处对应公式（4）
    #     self.estimated_weights = F.relu(output)
    #     # 对应公式（7）
    #     self.estimated_weights[self.estimated_weights < self.args.t_small] = 0.0

    #     return self.estimated_weights
    def get_estimated_weights(self, representations):
        x0 = representations[self.poten_edge_index[0]]
        x1 = representations[self.poten_edge_index[1]]
        output = torch.sum(x0 * x1, dim=1)
        estimated_weights = F.relu(output)
        estimated_weights = torch.where(
            estimated_weights < self.args.t_small,
            torch.zeros_like(estimated_weights),
            estimated_weights
        )
        return estimated_weights

    def reconstruct_loss(self, edge_index, representations):
        # 计算边预测器的损失函数
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=self.args.n_n * num_nodes)
        randn = randn[:, randn[0] < randn[1]]

        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)
        # 对应负样本损失公式
        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0], randn[1]] / self.args.sigma, 2)) @ F.mse_loss(neg,
                                                                                                                  torch.zeros_like(
                                                                                                                      neg),
                                                                                                                  reduction='none')
        # 对应正样本损失公式
        pos_loss = torch.exp(
            -torch.pow(self.features_diff[edge_index[0], edge_index[1]] / self.args.sigma, 2)) @ F.mse_loss(pos,
                                                                                                            torch.ones_like(
                                                                                                                pos),
                                                                                                            reduction='none')

        rec_loss = (pos_loss + neg_loss) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
def label_loss(edge_index, edge_weight, labels, idx_train, device):
    num_nodes = labels.shape[0]
    n_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    n_mask[idx_train]=1
    mask = n_mask[edge_index[0]]& n_mask[edge_index[1]]
    labeled_edge=edge_index[:,mask]
    labeled_weight = edge_weight[mask]

    #Y = F.softmax(labels,dim=1)
    Y = F.one_hot(labels)
    loss_label = ((labeled_weight+torch.pow(Y[labeled_edge[0]]-Y[labeled_edge[1]],2).sum(dim=1)-1)**2).sum()/num_nodes
    return loss_label