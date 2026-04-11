# This file is implemented based on this repository: https://github.com/levinhcntt/scAGCL

import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from model import HybridGATModel, SimpleGNNModel
from utils import load_h5_data1, load_h5_data2, GraphConstruction
from train import Train
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score,  normalized_mutual_info_score)
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA 
from GATEncoder import GATEncoderHybrid
from GNNEncoder import SimpleGNNEncoder
from torch_geometric.utils import add_self_loops
# from plyer import notification

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Pollen.h5')
    parser.add_argument('--num_cluster', type=int, default=8)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--gam', type=float, default=5.0)
    parser.add_argument('--subgraph_size', type=int, default=400) # 400
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=12345)   
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_epochs_ft', type=int, default=300)   
    parser.add_argument('--num_itersAdv', type=int, default=10)
    parser.add_argument('--attention_chunk_size', type=int, default=1024,
                        help='Process dense attention rows in blocks to reduce GPU memory. Set 0 to disable chunking.')
    args = parser.parse_args()

    config = {
        'num_hidden': 256, #256
        'num_proj_hidden': 256,
        'num_layers': 2,
        'weight_decay': 0.00001, # 0.00001
        'alpha': 100,
        'beta' : 0.01,
        'edge_r1': 0.4, # 0.4
        'edge_r2': 0.3, # 0.3
        'feature_r1': 0.3, # 0.3 
        'feature_r2': 0.4, # 0.4
        'tau': 0.5
    }
        
    num_hidden = config['num_hidden']
    num_proj_hidden =config['num_proj_hidden']
    num_layers = config['num_layers']
    weight_decay = config['weight_decay']
    alpha = config["alpha"] 
    beta = config["beta"] 
    edge_r1 = config['edge_r1']
    edge_r2 = config['edge_r2']
    feature_r1 = config['feature_r1']
    feature_r2 = config['feature_r2']
    tau=config['tau']

    data_file=args.data_file
    num_cluster=args.num_cluster
    lam = args.lam #adversarial weight
    gam = args.gam #finetuning weight
    subgraph_size=args.subgraph_size
    learning_rate=args.learning_rate

    num_epochs = args.num_epochs
    num_epochs_ft = args.num_epochs_ft
    num_itersAdv=args.num_itersAdv
    attention_chunk_size = args.attention_chunk_size
    random_seed=args.random_seed

    torch.manual_seed(random_seed)
    random.seed(random_seed) 
    np.random.seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)

    if torch.cuda.is_available():
        print('yeah use cuda!!')
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Read input data
    data_path='./data/' + data_file
    output_file = data_file + '_output.csv'
    X, Y = load_h5_data1(data_path)   # h5 file has X and Y field
    #X, Y = load_h5_data2(data_path)   #h5 file in which X in "exprs" and Y in "obs" field

    data = Data(x=torch.from_numpy(X))
    cellGraph=GraphConstruction(data, X, num_cluster)
    data.y=torch.tensor(Y, dtype=torch.int64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    encoder = GATEncoderHybrid(
        data.num_features,
        num_hidden,
        out_features=256,
        num_heads=4,
        dropout_prob=0.3,
    ).to(device)
    model = HybridGATModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    
    # encoder = SimpleGNNEncoder(
    #     data.num_features,
    #     num_hidden,
    #     out_features=256,
    # ).to(device)
    # model = SimpleGNNModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    
    #optimizer   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #Training

    # import pdb; pdb.set_trace()
    #Pre-train and fine-turning  
    with open("./result/" + output_file, "a") as f:
        print(f'=============================================', file=f)
        print(f'Run scGRAC model with: data: {data_file}, seed: {random_seed}, weight_decay: {weight_decay}', file=f)
        print(f'Version Not Topo-attack & ReLU activation', file=f)
        # print(f'Ablation study: SimpleGNNEncoder', file=f)
    
    embeddings, pred = Train(data_file, model, data, cellGraph, device, num_epochs,lam, num_cluster,alpha, beta, num_itersAdv, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components=None)

    X=data.x
    full_edgeind = data.edge_index.to(device)
    full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
    full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0]) # Dùng bản symmetric
    X = X.to(device)
    full_edgeind_self_loops.to(device)

    Z_pretrain = embeddings.clone()
    Z_pretrain = Z_pretrain.detach().cpu().numpy()
    Y=data.y
    Y = Y.detach().cpu().numpy()
    Z = normalize(Z_pretrain, norm='l2')
    kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=0)
    pred = kmeans.fit_predict(Z)

    ari_score = adjusted_rand_score(Y, pred)
    nmi_score = normalized_mutual_info_score(Y, pred)

    print('ARI score result: ', ari_score)
    print('NMI score result: ', nmi_score)