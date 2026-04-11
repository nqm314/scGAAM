# This file is implemented based on this repository: https://github.com/levinhcntt/scAGCL

from time import perf_counter as t
import numpy as np
import torch
import networkx as nx
from utils import EdgeDropping, GeneDropping, GraphAdversarialAttack, InitClusterCenters
# from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import add_self_loops, to_dense_adj
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans 
from torch.nn.functional import mse_loss
from torch.nn.functional import normalize as fnorm
import copy
import torch.nn.functional as F

torch.set_printoptions(threshold=float('inf'))

def Train(data_file, scAGCLmodel, data, cellGgraph, device, num_epochs, lam, num_cluster, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components):
    #For visualize with tensorboard
    # writer = SummaryWriter(log_dir="/content/drive/MyDrive/ĐACN-ĐATN/scAGCL/visualize_tensorboard")
    start_time = t()
    scAGCLmodel.train()   
   
    Z_final=None
    pred_final = None  
    bestloss=None

    X=data.x
    full_edgeind = data.edge_index.to(device)
    full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
    full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0]) # Dùng bản symmetric
    X = X.to(device)
    
    for epoch in range(1, num_epochs + 2): #do not use final round  
        scAGCLmodel.eval()
        Z_current = scAGCLmodel(X, full_edgeind_self_loops, None) #get embeddings
        scAGCLmodel.train() #back to continue training

        subGraph = cellGgraph.subgraph(np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size])
        x_sub = data.x[np.array(subGraph.nodes())].to(device)
        subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
        edgeind = np.array(subGraph .edges()).T
        edgeind = torch.from_numpy(edgeind).to(device).long()
        edgeind_symmetric = torch.cat([edgeind, edgeind.flip(0)], dim=1)
        edgeind_self_loops, _ = add_self_loops(edgeind_symmetric, num_nodes = x_sub.shape[0])
        adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
        adj = torch.clamp(adj, 0, 1)
        assert torch.equal(adj, adj.transpose(0,1))

        optimizer.zero_grad()
        x_1 = GeneDropping(x_sub, feature_r1)
        x_2 = GeneDropping(x_sub, feature_r2)

        edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
        edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
        edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
        edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])

        adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
        adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
        adj_1 = torch.clamp(adj_1, 0,1)
        adj_2 = torch.clamp(adj_2, 0,1)

        z_1 = scAGCLmodel(x_1, adj_1)
        z_2 = scAGCLmodel(x_2, adj_2)   
        loss1= scAGCLmodel.loss(z_1,z_2,batch_size=0) 
        
        loss2=0
        if lam > 0:
            adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
            adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
            z_3 = scAGCLmodel(x_3, adj_3)
            z_4 = scAGCLmodel(x_4, adj_4)
            loss2 = scAGCLmodel.loss(z_3,z_4,batch_size=0)
        
        loss = loss1 + lam*loss2

        # Modified
        # with torch.autograd.detect_anomaly(): 
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(scAGCLmodel.parameters(), max_norm=0.1)
        # Modified
        optimizer.step()

        # logging
        Z_eval = Z_current.clone()
        Y=data.y
        Z = Z_eval.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        Z = normalize(Z, norm='l2')
        kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=0)
        pred = kmeans.fit_predict(Z)
        
        now_time = t()
        if epoch <= num_epochs and epoch%10==0:
            # import pdb; pdb.set_trace()
            print(f'Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}, total loss={loss:.4f}, total time {now_time - start_time:.4f}')

        if epoch == 1:
            bestloss = loss
            Z_final=Z_current
            pred_final = pred

        if loss < bestloss:
            bestloss = loss
            Z_final=Z_current
            pred_final = pred 

    return Z_final, pred_final