import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from model import NewModel2, Finetune_Model, HybridGATModel, HybridGATFinetune_Model, UnifiedGATModel, NewModel, SimpleGNNModel
from scGRAC.utils import load_h5_data1, load_h5_data2, GraphConstruction
from train import Train, Finetuning
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score,  normalized_mutual_info_score)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA 
from GATEncoder import GATEncoder, GATEncoderDense, GATEncoderHybrid, UnifiedGATEncoder
from GNNEncoder import SimpleGNNEncoder
from torch_geometric.utils import add_self_loops
# from plyer import notification

import warnings
warnings.filterwarnings("ignore")

import sys
import psutil
import time
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Pollen')
    parser.add_argument('--num_cluster', type=int, default=8)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--gam', type=float, default=5.0)
    parser.add_argument('--subgraph_size', type=int, default=400) # 400
    parser.add_argument('--learning_rate', type=float, default=0.0005)
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

    # --- SETUP CSV ---
    filename_csv = './results_scGRAC.csv'
    fieldnames =['Dataset', '1', '11', '111', '1111', '11111', 'Total']
    all_data =[]
    
    if os.path.isfile(filename_csv):
        with open(filename_csv, 'r', newline='') as f_read:
            reader = csv.DictReader(f_read)
            for row in reader:
                all_data.append(row)

    process = psutil.Process(os.getpid())

    f_datasets = open("./data1.txt", 'r')
    for line in f_datasets:
        parts = line.split()
        if not parts: continue
        filetype, data_file = parts
        
        # # Check if this dataset already exists
        # check_name = f"{data_file} (ARI)"
        # if any(row['Dataset'] == check_name for row in all_data):
        #     print(f"... Skipping {data_file}, record already exists ...\n")
        #     continue

        print(f"\n{'='*40}")
        print(f"====== Processing Dataset: {data_file} ======")
        print(f"{'='*40}")

        # Read input data
        data_path = f'D:/scRNA-sq_Datasets/h5data{filetype}/{data_file}.h5'
        if filetype == "1":
            X_raw, Y_raw = load_h5_data1(data_path)   
        else:
            X_raw, Y_raw = load_h5_data2(data_path)

        num_cluster = len(np.unique(Y_raw))
        print(f"Dataset shape: {X_raw.shape}, Clusters: {num_cluster}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True) 
            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # # Read input data
        # data_path = 'D:/scRNA-sq_Datasets/h5data2/' + data_file + '.h5'
        output_file = data_file + '_output.csv'
        # X, Y = load_h5_data1(data_path)   # h5 file has X and Y field
        # #X, Y = load_h5_data2(data_path)   #h5 file in which X in "exprs" and Y in "obs" field

        # Lists for tracking metrics
        ari_list, nmi_list, time_list, gpu_list, ram_list = [], [], [], [], []

        # for random_seed in [1, 11, 111, 1111, 11111]:
        for random_seed in [1]:

            torch.manual_seed(random_seed)
            random.seed(random_seed) 
            np.random.seed(random_seed)

            X = X_raw.copy()
            Y = Y_raw.copy()

            # Reset GPU memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            seed_start_time = time.time()

            data = Data(x=torch.from_numpy(X))
            cellGraph=GraphConstruction(data, X, num_cluster)
            data.y=torch.tensor(Y, dtype=torch.int64)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

            encoder = GATEncoderHybrid(
                data.num_features,
                num_hidden,
                out_features=256,
                num_heads=4,
                dropout_prob_1=0.3,
                dropout_prob_2=0.3,
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
            
            # with open("./result/" + output_file, "a") as f:
            #     print(f'=============================================', file=f)
            #     print(f'Run scGRAC model with: data: {data_file}, seed: {random_seed}, weight_decay: {weight_decay}', file=f)
            #     print(f'Version Not Topo-attack & ReLU activation & lam = 0', file=f)
            
            embeddings, pred, ari, nmi, model_state = Train(data_file, model, data, cellGraph, device, num_epochs,lam, num_cluster,alpha, beta, num_itersAdv, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components=None)
                
            if model_state is not None:
                print("Loading best model state from pre-training...")
                model.load_state_dict(model_state)
                model.eval()
            else:
                print("Warning: best_state is None. Using final model state.")

            X=data.x
            # Adj = edge2adj(X, data.edge_index).to_dense()
            full_edgeind = data.edge_index.to(device)
                    # full_edgeind_self_loops, _ = add_self_loops(full_edgeind, num_nodes = X.shape[0])
            full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
            full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0]) # Dùng bản symmetric
                    # Adj = to_dense_adj(full_edgeind_self_loops, max_num_nodes= X.shape[0])[0]
                    # Adj = torch.clamp(Adj, 0, 1)
            X = X.to(device)
                    # Adj = Adj.to(device)
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
            sil_max = silhouette_score(normalize(Z_pretrain, norm='l2'), pred)
            print('ARI score for pretrain result: ', ari_score)
            print('NMI score for pretrain result: ', nmi_score)
            print('Silhouette score for pretrain result: ', sil_max)

            seed_training_time = time.time() - seed_start_time
            sys_ram_gb = process.memory_info().rss / (1024**3)
            
            peak_vram_gb = 0.0
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

            # Append metrics
            ari_list.append(ari_score)
            nmi_list.append(nmi_score)
            time_list.append(seed_training_time)
            gpu_list.append(peak_vram_gb)
            ram_list.append(sys_ram_gb)

            print(f">> Seed {random_seed} Results:")
            print(f"   - ARI:  {ari_score:.4f}")
            print(f"   - NMI:  {nmi_score:.4f}")
            print(f"   - Time: {seed_training_time:.2f} s")
            print(f"   - GPU:  {peak_vram_gb:.3f} GB")
            print(f"   - RAM:  {sys_ram_gb:.2f} GB")
            print("-" * 50)

            ################################## Drawing UMAP
            print("\n--- Generating UMAP ---")
            final_z = Z_pretrain
            ground_truth = Y
            predicted_labels = pred

            print("Calculating UMAP 2D projection...")
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
            embedding_2d = reducer.fit_transform(final_z)

            # Setup Plotting (1 Row, 2 Columns)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))
            
            # Define color palette based on number of clusters
            num_classes = len(np.unique(ground_truth))
            palette = sns.color_palette("tab20", num_classes) if num_classes > 10 else sns.color_palette("tab10", num_classes)

            # --- Plot 1: Model Predictions ---
            sns.scatterplot(
                x=embedding_2d[:, 0], 
                y=embedding_2d[:, 1], 
                hue=predicted_labels, 
                palette=palette, 
                s=20, # dot size
                alpha=0.8,
                ax=ax1,
                legend=False # Hide legend here to save space
            )
            ax1.set_title('Full model (Predicted)', fontsize=18, fontweight='bold', pad=12)
            
            # Remove axis ticks and borders to look clean like the reference image
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')

            # --- Plot 2: Ground Truth ---
            sns.scatterplot(
                x=embedding_2d[:, 0], 
                y=embedding_2d[:, 1], 
                hue=ground_truth, 
                palette=palette, 
                s=20, # dot size
                alpha=0.8,
                ax=ax2,
                legend=False
            )
            ax2.set_title(r'Ground truth (Full model)', fontsize=18, fontweight='bold', pad=12)
            
            # Remove axis ticks to look clean
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlabel('')
            ax2.set_ylabel('')
            for spine in ax2.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')

            # # Adjust legend position for the second plot
            # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cell Types", fontsize=10, title_fontsize=12)

            # Save and Show
            plt.subplots_adjust(hspace=0.2)
            # Ensure directory exists
            umap_dir = "./visualization/umap/"
            if not os.path.exists(umap_dir):
                os.makedirs(umap_dir)
                
            save_name = os.path.join(umap_dir, f"umap_{data_file}.png")
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"UMAP saved successfully to: {save_name}")
            ################################################# END OF UMAP

            torch.cuda.empty_cache()
                
            # finetune_model = HybridGATFinetune_Model(model, num_hidden, num_proj_hidden, num_cluster, 1, tau, tau_proto=0.3)
            # finetune_lr = learning_rate
            # optimizer1 = torch.optim.Adam(finetune_model.parameters(), lr=finetune_lr, weight_decay = weight_decay)

            # alpha_ft = 100
            # edge_r1_ft = 0.4 # 0.4
            # edge_r2_ft = 0.3 # 0.3
            # feature_r1_ft = 0.3 # 0.3 
            # feature_r2_ft = 0.4 # 0.4
            # Finetuning(finetune_model, embeddings, sil_max, ari_score, nmi_score, data, cellGraph, device, num_epochs_ft, lam, gam, num_cluster, alpha_ft, beta, num_itersAdv, optimizer1, edge_r1_ft, edge_r2_ft, feature_r1_ft, feature_r2_ft, subgraph_size, output_file, principal_components=None)

        # --- PREPARE NEW ROWS AND UPDATE CSV ---
        avg_ari = sum(ari_list) / len(ari_list)
        avg_nmi = sum(nmi_list) / len(nmi_list)
        avg_time = sum(time_list) / len(time_list)
        avg_gpu = sum(gpu_list) / len(gpu_list)
        avg_ram = sum(ram_list) / len(ram_list)
        
        def create_row(metric_name, data_list, average):
            return {
                'Dataset': metric_name,
                '1': data_list[0], '11': data_list[1], '111': data_list[2], 
                '1111': data_list[3], '11111': data_list[4], 
                'Total': average
            }

        all_data.extend([
            create_row(f"{data_file} (ARI)", ari_list, avg_ari),
            create_row(f"{data_file} (NMI)", nmi_list, avg_nmi),
            create_row(f"{data_file} (Time s)", time_list, avg_time),
            create_row(f"{data_file} (GPU GB)", gpu_list, avg_gpu),
            create_row(f"{data_file} (RAM GB)", ram_list, avg_ram)
        ])
        
        with open(filename_csv, 'w', newline='') as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=fieldnames)
            logwriter.writeheader()
            logwriter.writerows(all_data)

        print(f"\nCSV file successfully updated for {data_file}!")