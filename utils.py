#Some codes are modified from https://github.com/Shengyu-Feng/ARIEL, https://github.com/xuebaliang/scziDesk

from __future__ import print_function

from typing import Tuple
import networkx as nx
import torch
from torch import Tensor
import h5py
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize as skl_normalize
from torch_scatter import scatter_add
from torch_geometric.utils import  add_self_loops, dense_to_sparse, to_dense_adj
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def GraphConstruction(data, features, num_clusters):
    cell_num = features.shape[0]
    average_num = cell_num // num_clusters
    neighbor_num = average_num // 10
    neighbor_num = min(neighbor_num, 15)
    neighbor_num = max(neighbor_num, 5)
    print("Number of neighbors (k) in k-nn graph: ", neighbor_num)
 
    # Calculate Pearson distance matrix
    dis_matrix = squareform(pdist(features, metric='correlation'))
    # print("Pearson distance matrix: ", dis_matrix)
    print("Pearson distance matrix dimensions: ", dis_matrix.shape)
    # Build kNN graph
    print("Start building kNN graph !!!")
    nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric='precomputed').fit(dis_matrix)
    print("End building kNN graph !!!")
    # _, indices = nbrs.kneighbors(dis_matrix)  # Get only indices
    dists, indices = nbrs.kneighbors(dis_matrix)

    source_nodes = np.repeat(np.arange(cell_num), neighbor_num)
    target_nodes = indices.flatten()
    # Tính similarity làm trọng số (quan trọng cho Soft-Attack của cậu)
    edge_weights = 1.0 - dists.flatten()

    # Chuyển sang Tensor
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    adj_mat = np.zeros((cell_num, cell_num))
    for i in range(cell_num*neighbor_num):
        adj_mat[edge_index[0, i], edge_index[1, i]] = 1
 
    # Create adjacency matrix
    n_samples = features.shape[0]
    adj_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in indices[i]:
             adj_matrix[i, j] = 1  

    if not np.array_equal(adj_mat, adj_matrix):
        print("Different adj_mat and adj_matrix")
        print("Adj_mat: ", adj_mat)
        print("Adj_matrix: ", adj_matrix)

    #Create a list of egdes
    edge_list = torch.empty((2,0), dtype=torch.int64)
    for i in range( adj_mat.shape[0]):
     for j in range(i + 1,   adj_mat.shape[1]):
        if  adj_mat[i, j] == 1:
            col = torch.tensor([i, j], dtype=torch.int64)
            edge_list = torch.cat((edge_list, col.unsqueeze(1)), dim=1)

    # print("Edge index.shape: ", edge_index.shape)
    # print("Edge list.shape: ", edge_list.shape)
    # if edge_index != edge_list:
    #     print("Different edge_index and edge_list")

    data.edge_index=edge_list
    # data.edge_index=edge_index
          
    #generate a graph
    G = nx.from_numpy_array(adj_mat)
    print("Building a " + str(G))
    print("===================================================")

    return G

#------------------------------------------------------------------
def normalize_adj_tensor(adj):
    """Symmetrically normalize adjacency tensor."""
    rowsum = torch.sum(adj,1) + 1e-8
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def normalize_adj_tensor_sp(adj):
    """Symmetrically normalize sparse adjacency tensor."""
    device = adj.device
    adj = adj.to("cpu")
    rowsum = torch.spmm(adj, torch.ones((adj.size(0),1))).reshape(-1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = torch.mm(torch.smm(adj.transpose(0,1),d_mat_inv_sqrt.transpose(0,1)),d_mat_inv_sqrt)
    return adj.to(device)

def edge2adj(x, edge_index):
    """Convert edge index to adjacency matrix"""
    num_nodes = x.shape[0]
    tmp, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(tmp.size(1), dtype=None,
                                     device=edge_index.device)
    row, col = tmp[0], tmp[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    #return torch.sparse.FloatTensor(tmp, edge_weight,torch.Size((num_nodes, num_nodes)))
    return torch.sparse_coo_tensor(tmp, edge_weight, torch.Size((num_nodes, num_nodes)))

def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix


#---------------Reading files .h file 1----------------------------------------#

"""
Load scRNA-seq data set from .h5 file and perfrom preprocessing
"""
def load_h5_data1(data_path):
    print("Reading data!")
    inputFile = h5py.File(data_path, 'r')
    if 'obs' in inputFile:
      mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
       
      if isinstance(mat, np.ndarray):
          X = np.array(mat)
      else:
          X = np.array(mat.toarray())
      cell_name = np.array(obs["cell_type1"])

      _, Y = np.unique(cell_name, return_inverse=True)
      X = X.astype('float32')
    
    else:
      X = np.array(inputFile['X']).astype('float32')
      Y = np.array(inputFile['Y'])

      if Y.dtype != "int64":
          encoder_x = LabelEncoder()
          Y = encoder_x.fit_transform(Y)
      inputFile.close()
      
    X = preprocess(X, nb_genes=2000)
    return X, Y


def preprocess(X, nb_genes = 2000):
    """
    Preprocessing phase as proposed in scanpy package.
    Keeps only nb_genes most variable genes and normalizes
    the data to 0 mean and 1 std.
    Args:
        X ([type]): [description]
        nb_genes (int, optional): [description]. Defaults to 500.
    Returns:
        [type]: [description]
    """

    print("There are " + str(X.shape[0]) + " cells, " + str(X.shape[1]) + " genes")

    adata = sc.AnnData(X)
    adata = normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X.astype('float32')
    print(f"Keeping {nb_genes} genes")
    return X


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, 
              size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes 
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=1)
        #sc.pp.filter_cells(adata, min_genes=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    return adata

#---------------Reading files .h file 2----------------------------------------#
def load_h5_data2(data_path, is_NE=False, n_clusters=20, K=None):
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
       
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])

    _, Y = np.unique(cell_name, return_inverse=True)
    
    X =X.astype('float32')
    X=preprocess(X, nb_genes=2000)
    return X, Y

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

#-----------------------------------------------------------------------------
#Graph augmentation

def EdgeDropping(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.bernoulli(torch.ones(row.size(0), device=edge_index.device) * (1 - p)).type(torch.bool)

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def GeneDropping(x, drop_prob):
    drop_mask = torch.bernoulli(torch.ones(x.size(1), device=x.device) * (1 - drop_prob)).type(torch.bool) 
    x = x.clone()
    x[:, drop_mask] = 0
    return x


#-----------------------------------------------------------------------------
class GCNConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
#-----------------------------------------------------------------------------
# def GraphAdversarialAttack(model, adj_sub, adj_aug, x_sub, x_aug, iters, node_ratio, alpha, beta, principal_components):
#     """ PGD attack on both features and edges"""

#     for param in  model.parameters():
#         param.requires_grad = False
#     model.eval()
#     device = x_sub.device
#     total_edges = torch.sum(adj_aug)         
#     n_node = x_aug.shape[0]
#     eps = total_edges * node_ratio/2
#     xi = 1e-3
    
#     A_ = adj_aug   
#     C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
#     S_ = torch.zeros_like(A_, requires_grad= True)
    
#     mask = torch.ones_like(A_)
#     mask = mask - torch.tril(mask)

#     delta = torch.zeros_like(x_aug, device=device, requires_grad=True)

#     # if principal_components is not None:
#     #   rand_coeffs = torch.randn(x_2.shape[0], principal_components.shape[0], device=device)
#     #   delta_init = torch.matmul(rand_coeffs, principal_components)
#     #   delta_init = delta_init.sign() * 0.04
#     #   delta = delta_init.clone().detach().requires_grad_(True)
      
#     # adj_1 = edge2adj(x_1, edge_index_1)
#     model.to(device)
#     # adj_sub_edge_ind = dense_to_sparse(adj_sub)

#     # discretized_S_ = torch.zeros_like(S_)

#     for itr in range(iters):
#         S = (S_ * mask)
#         S = S + S.T
#         # S = S_
#         A_prime = A_ + (S * C_)
#         # A_prime = torch.clamp(A_prime, min = 0)
#         adj_hat = A_prime + torch.eye(n_node,device=device)
#         adj_hat_clamped = torch.clamp(adj_hat, 0, 1)
#         # adj_hat_clamped = (adj_hat_clamped + adj_hat_clamped.t()) / 2
#         # check_is_strictly_binary(adj_hat_clamped)
#         # inspect_fractional_values(adj_hat_clamped)
#         # adj_hat_edge_ind = dense_to_sparse(adj_hat_clamped)
#         # sub_edge_index, sub_edge_weight = dense_to_differentiable_sparse(adj_sub)
#         # z1 = model(x_sub, sub_edge_index, sub_edge_weight)
#         assert torch.equal(adj_sub, adj_sub.transpose(0,1))
#         z1 = model(x_sub, adj_sub)
#         # print(adj_hat_clamped)
#         # attk_edgeind, attk_edge_weight = dense_to_differentiable_sparse(adj_hat_clamped)
#         # z2 = model(x_aug + delta, attk_edgeind, attk_edge_weight) 
#         # z2 = model(x_aug + delta, adj_hat_clamped, None)
#         # assert torch.equal(adj_hat_clamped, adj_hat_clamped.transpose(0,1))
#         z2 = model(x_aug + delta, adj_hat_clamped)
#         Attackloss = model.loss(z1, z2, batch_size=0) 
#         Attackloss.backward()
#         # import pdb; pdb.set_trace()
#         # Modified
#         torch.nn.utils.clip_grad_norm_(S_, max_norm=0.5)
#         torch.nn.utils.clip_grad_norm_(delta, max_norm=0.5)
#         # Modified
#         S_.data = (S_.data + alpha/np.sqrt(itr+1)*S_.grad.detach()) # annealing
#         S_.data = bisection(S_.data, eps, xi) # clip S
#         S_.grad.zero_()
        
#         delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
#         delta.grad.zero_()

#         randm = torch.rand(n_node, n_node,device=device)
#         discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
#         discretized_S = discretized_S * mask 
#         discretized_S = discretized_S + discretized_S.T
#         A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
#         # check_is_strictly_binary(A_hat)
        
#     for param in model.parameters():
#         param.requires_grad = True
#     model.train()
#     x_hat = x_aug + delta.data.to(device)
#     # A_hat_clamped = A_hat + torch.eye(n_node,device=device)
#     A_hat_clamped = torch.clamp(A_hat, 0, 1)
#     check_is_strictly_binary(A_hat_clamped)
#     # inspect_fractional_values(A_hat_clamped)
#     # assert torch.equal(A_hat_clamped, A_hat_clamped.transpose(0,1))
#     return A_hat_clamped, x_hat

def GraphAdversarialAttack(model, adj_sub, adj_aug, x_sub, x_aug, iters, node_ratio, alpha, beta, principal_components):
    """ PGD attack on both features and edges"""

    for param in  model.parameters():
        param.requires_grad = False
    model.eval()
    device = x_sub.device
    total_edges = torch.sum(adj_aug)         
    n_node = x_aug.shape[0]
    eps = total_edges * node_ratio/2
    xi = 1e-3
    
    # A_ = adj_aug   
    # C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
    # S_ = torch.zeros_like(A_, requires_grad= True)
    
    # mask = torch.ones_like(A_)
    # mask = mask - torch.tril(mask)

    delta = torch.zeros_like(x_aug, device=device, requires_grad=True)

    # if principal_components is not None:
    #   rand_coeffs = torch.randn(x_2.shape[0], principal_components.shape[0], device=device)
    #   delta_init = torch.matmul(rand_coeffs, principal_components)
    #   delta_init = delta_init.sign() * 0.04
    #   delta = delta_init.clone().detach().requires_grad_(True)
      
    # adj_1 = edge2adj(x_1, edge_index_1)
    model.to(device)
    # adj_sub_edge_ind = dense_to_sparse(adj_sub)

    # discretized_S_ = torch.zeros_like(S_)

    for itr in range(iters):
        # S = (S_ * mask)
        # S = S + S.T
        # # S = S_
        # A_prime = A_ + (S * C_)
        # A_prime = torch.clamp(A_prime, min = 0)
        # adj_hat = A_prime + torch.eye(n_node,device=device)
        # adj_hat_clamped = torch.clamp(adj_hat, 0, 1)
        # adj_hat_clamped = (adj_hat_clamped + adj_hat_clamped.t()) / 2
        # check_is_strictly_binary(adj_hat_clamped)
        # inspect_fractional_values(adj_hat_clamped)
        # adj_hat_edge_ind = dense_to_sparse(adj_hat_clamped)
        # sub_edge_index, sub_edge_weight = dense_to_differentiable_sparse(adj_sub)
        # z1 = model(x_sub, sub_edge_index, sub_edge_weight)
        z1 = model(x_sub, adj_sub)
        # print(adj_hat_clamped)
        # attk_edgeind, attk_edge_weight = dense_to_differentiable_sparse(adj_hat_clamped)
        # z2 = model(x_aug + delta, attk_edgeind, attk_edge_weight) 
        # z2 = model(x_aug + delta, adj_hat_clamped, None)
        # z2 = model(x_aug + delta, adj_hat_clamped)
        z2 = model(x_aug + delta, adj_aug)
        Attackloss = model.loss(z1, z2, batch_size=0) 
        Attackloss.backward()
        # import pdb; pdb.set_trace()
        # Modified
        # torch.nn.utils.clip_grad_norm_(S_, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(delta, max_norm=0.5)
        # Modified
        # S_.data = (S_.data + alpha/np.sqrt(itr+1)*S_.grad.detach()) # annealing
        # S_.data = bisection(S_.data, eps, xi) # clip S
        # S_.grad.zero_()
        
        delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
        delta.grad.zero_()

        # randm = torch.rand(n_node, n_node,device=device)
        # discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
        # discretized_S = discretized_S + discretized_S.T
        # A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
        # check_is_strictly_binary(A_hat)
        
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    x_hat = x_aug + delta.data.to(device)
    # assert torch.equal(A_hat, A_hat.transpose(0,1))
    # A_hat_clamped = A_hat + torch.eye(n_node,device=device)
    # A_hat_clamped = torch.clamp(A_hat_clamped, 0, 1)
    # check_is_strictly_binary(A_hat_clamped)
    # inspect_fractional_values(A_hat_clamped)
    adj_aug = torch.clamp(adj_aug, 0, 1)
    return adj_aug, x_hat
    # return A_hat_clamped, x_hat
    
def bisection(a,eps,xi,ub=1):
    pa = torch.clamp(a, 0, ub)
    if torch.sum(pa) <= eps:
        upper_S_update = pa
    else:
        mu_l = torch.min(a-1)
        mu_u = torch.max(a)
        mu_a = (mu_u + mu_l)/2
        while torch.abs(mu_u - mu_l)>xi:
            mu_a = (mu_u + mu_l)/2
            gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
            gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
            if gu == 0:
                break
            if torch.sign(gu) == torch.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = torch.clamp(a-mu_a, 0, ub)
    return upper_S_update


def InitClusterCenters(embedding, num_cluster, device):
    # 1. Fit Spectral Clustering using scikit-learn
    Z = embedding.clone()
    Z = Z.detach().cpu().numpy()
    # Z = skl_normalize(Z, norm='l2')
    #clustering = SpectralClustering(n_clusters=num_cluster, affinity='rbf', random_state=0)
    clustering = KMeans(n_clusters=num_cluster, init="k-means++", n_init=20, random_state=0)
    clustering.fit_predict(Z) 
    
    cluster_centers_np = clustering.cluster_centers_ 

    cluster_centers = torch.from_numpy(cluster_centers_np).float().to(device)

    # cluster_centers = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)

    # # 2. Convert cluster assignments back to PyTorch tensor
    # cluster_assignments = torch.tensor(cluster_assignments_np, device=device)

    # # 3. Calculate cluster centers (using detached embeddings)
    # cluster_centers = []
    # for i in range(num_cluster):
    #     cluster_indices = torch.where(cluster_assignments == i)[0]
    #     cluster_center = embedding.detach()[cluster_indices].mean(dim=0) # Detach here as well
    #     cluster_centers.append(cluster_center)

    # # 4. Stack cluster centers into a tensor
    # cluster_centers = torch.stack(cluster_centers, dim=0).to(device)

    return cluster_centers

def check_is_strictly_binary(matrix):
    # Lấy các giá trị duy nhất trong ma trận
    unique_vals = torch.unique(matrix)
    
    # print(f"--- STRICT CHECK ---")
    # print(f"Các giá trị duy nhất trong ma trận: {unique_vals}")
    
    # Kiểm tra xem tất cả phần tử có phải là 0 HOẶC 1 không
    is_binary = torch.all((matrix == 0) | (matrix == 1))
    
    if not is_binary:
        print("KẾT LUẬN: Ma trận KHÔNG PHẢI nhị phân (chứa số thực ở giữa).")
        # In ra số lượng phần tử "lai tạp"
        fractional_count = ((matrix > 0) & (matrix < 1) | (matrix > 1)).sum().item()
        print(f"Số lượng phần tử khác 0 và 1: {fractional_count}")
    # print("--------------------")


def inspect_fractional_values(matrix):
    # Lọc ra các giá trị > 0 và < 1 (không phải 0, cũng chẳng phải 1)
    mask = (matrix > 0) & (matrix < 1) & (matrix > 1)
    fractional_values = matrix[mask]
    
    if fractional_values.numel() > 0:
        print(f"--- PGD REALITY CHECK ---")
        print(f"Phát hiện {fractional_values.numel()} cạnh có trọng số thực (weighted edges)!")
        print(f"Min (khác 0): {fractional_values.min().item():.6f}")
        print(f"Max (khác 1): {fractional_values.max().item():.6f}")
        print(f"Mean: {fractional_values.mean().item():.6f}")
        print(f"10 giá trị mẫu ngẫu nhiên: {fractional_values[:10].tolist()}")
        print("-------------------------")
    else:
        print("Ma trận sạch, không có giá trị phân số.")

def dense_to_differentiable_sparse(dense_adj):
    device = dense_adj.device
    N = dense_adj.size(0)

    rows = torch.arange(N, device=device).repeat_interleave(N)
    cols = torch.arange(N, device=device).repeat(N)
    edge_ind = torch.stack([rows, cols], dim=0)

    edge_weight = dense_adj.contiguous().view(-1)
    return edge_ind, edge_weight