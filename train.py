from time import perf_counter as t
import numpy as np
import torch
import networkx as nx
from utils_run import edge2adj, EdgeDropping, GeneDropping, GraphAdversarialAttack, InitClusterCenters, check_is_strictly_binary, inspect_fractional_values, dense_to_differentiable_sparse
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
    best_model_state = None 
    bessloss=None

    ari_final, nmi_final = 0, 0 
    silh_score = 0.0
    sil_res = []
    sil_array = None

    X=data.x
    # Adj = edge2adj(X, data.edge_index).to_dense()
    full_edgeind = data.edge_index.to(device)
    # full_edgeind_self_loops, _ = add_self_loops(full_edgeind, num_nodes = X.shape[0])
    full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
    full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0]) # Dùng bản symmetric
    # full_edgeind_self_loops = full_edgeind_symmetric
    Adj = to_dense_adj(full_edgeind_self_loops, max_num_nodes= X.shape[0])[0]
    Adj = torch.clamp(Adj, 0, 1)
    # full_edgeind, full_edge_weight = dense_to_differentiable_sparse(Adj)
    X = X.to(device)
    Adj = Adj.to(device)
    # full_edgeind_self_loops = full_edgeind_symmetric
    # full_edgeind_self_loops.to(device)
    
    for epoch in range(1, num_epochs + 2): #do not use final round  
        # import pdb; pdb.set_trace()     
        scAGCLmodel.eval()
        Z_current = scAGCLmodel(X, full_edgeind_self_loops, None) #get embeddings
        # Z_current = scAGCLmodel(X, Adj)
        # assert torch.equal(Adj, Adj.transpose(0,1))
        # Z_current_dense = scAGCLmodel(X, Adj) 
        # diff = torch.abs(Z_current - Z_current_dense).max().item()
        # print(f"Max Difference: {diff:.8f}")
        
        # # Đại ca cho phép sai số < 1e-5 (do khác biệt thuật toán cộng float)
        # if diff < 1e-5:
        #     print("✅ TEST PASSED: CHÚC MỪNG EM! Hai chế độ đã đồng nhất.")
        # else:
        #     print("❌ TEST FAILED.")
        #     print("Dense Sample:", Z_current_dense[0, :3])
        #     print("Sparse Sample:", Z_current[0, :3])  

        state = copy.deepcopy(scAGCLmodel.state_dict())
        # with open("./result/" + output_file, "a") as f:
        #     print(f"Epoch: {epoch}, State: {state}", file=f) 
        # Z_current = scAGCLmodel(X, Adj)
        scAGCLmodel.train() #back to continue training

        subGraph = cellGgraph.subgraph(np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size])
        x_sub = data.x[np.array(subGraph.nodes())].to(device)
        subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
        edgeind = np.array(subGraph .edges()).T
        # edgeind = torch.LongTensor(np.hstack([edgeind,edgeind[::-1]])).to(device)
        edgeind = torch.from_numpy(edgeind).to(device).long()
        edgeind_symmetric = torch.cat([edgeind, edgeind.flip(0)], dim=1)
        edgeind_self_loops, _ = add_self_loops(edgeind_symmetric, num_nodes = x_sub.shape[0])
        # edgeind_self_loops = edgeind_symmetric
        # edgeind_self_loops, _ = add_self_loops(edgeind, num_nodes = x_sub.shape[0])
        adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
        adj = torch.clamp(adj, 0, 1)
        assert torch.equal(adj, adj.transpose(0,1))

        optimizer.zero_grad()
        x_1 = GeneDropping(x_sub, feature_r1)
        x_2 = GeneDropping(x_sub, feature_r2)
        # x_1 = GeneDropping(X, feature_r1)
        # x_2 = GeneDropping(X, feature_r2)

        edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
        edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
        # edgeind_1 = EdgeDropping(full_edgeind_self_loops, p=edge_r1, force_undirected=True)[0]
        # edgeind_2 = EdgeDropping(full_edgeind_self_loops, p=edge_r2, force_undirected=True)[0]
        # adj_1 = edge2adj(x_1, edgeind_1).to_dense()
        # adj_2 = edge2adj(x_2, edgeind_2).to_dense()
        edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
        edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])
        # edgeind_1_self_loops = edgeind_1
        # edgeind_2_self_loops = edgeind_2

        adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
        adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
        adj_1 = torch.clamp(adj_1, 0,1)
        adj_2 = torch.clamp(adj_2, 0,1)
        assert torch.equal(adj_1, adj_1.transpose(0,1))
        assert torch.equal(adj_2, adj_2.transpose(0,1))
        # check_is_strictly_binary(adj_1)
        # inspect_fractional_values(adj_1)
        # check_is_strictly_binary(adj_2)
        # inspect_fractional_values(adj_2)

        z_1 = scAGCLmodel(x_1, adj_1)
        z_2 = scAGCLmodel(x_2, adj_2)   
        # z_1 = scAGCLmodel(x_1, adj_1, None)
        # z_2 = scAGCLmodel(x_2, adj_2, None)   
        # z_1 = scAGCLmodel(x_1, edgeind_1_self_loops, None)
        # z_2 = scAGCLmodel(x_2, edgeind_2_self_loops, None)
        loss1= scAGCLmodel.loss(z_1,z_2,batch_size=0) 
        
        loss2=0
        if lam > 0:
            adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
            adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
            # adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, Adj, adj_1, X, x_1, iters, 0.2, alpha, beta, principal_components)
            # adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, Adj, adj_2, X, x_2, iters, 0.2, alpha, beta, principal_components)
            # adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, edgeind_self_loops, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
            # adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, edgeind_self_loops, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
            # edgeind_3, edge_weight_3 = dense_to_differentiable_sparse(adj_3)
            # z_3 = scAGCLmodel(x_3, edgeind_3, edge_weight_3)
            # z_3 = scAGCLmodel(x_3, adj_3, None)
            z_3 = scAGCLmodel(x_3, adj_3)
            # edgeind_4, edge_weight_4 = dense_to_differentiable_sparse(adj_4)
            # z_4 = scAGCLmodel(x_4, edgeind_4, edge_weight_4)
            # z_4 = scAGCLmodel(x_4, adj_4, None)
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

        ari_score = adjusted_rand_score(Y, pred)
        nmi_score = normalized_mutual_info_score(Y, pred)

        if len(np.unique(pred)) > 1:
            silh_score = silhouette_score(Z, pred)
        else:
            silh_score = -1.0 

        # --- TensorBoard logging ---
        # writer.add_scalar('Loss/total', loss.item(), epoch)
        # writer.add_scalar('Loss/loss1', loss1.item(), epoch)
        # writer.add_scalar('Loss/loss2', loss2.item(), epoch)
        # writer.add_scalar("Clustering/ARI", ari_score, epoch)
        # writer.add_scalar("Clustering/NMI", nmi_score, epoch)
        
        now_time = t()
        if epoch <= num_epochs and epoch%10==0:
            # import pdb; pdb.set_trace()
            print(f'Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}, total loss={loss:.4f}, total time {now_time - start_time:.4f}')
            # print(f'ARI={ari_score:.4f}, NMI={nmi_score:.4f}, sil_score={silh_score:.4f}')
        
        # with open("./result/" + output_file, "a") as f:
        #     print(f'Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}, total loss={loss:.4f}, total time {now_time - start_time:.4f}', file=f)
        #     print(f'ARI={ari_score:.4f}, NMI={nmi_score:.4f}, sil_score={silh_score:.4f}', file=f)

        if epoch == 1:
            bessloss = loss
            sil_final = silh_score
            Z_final=Z_current
            pred_final = pred
            # best_model_state = copy.deepcopy(scAGCLmodel.state_dict())
            best_model_state = state

        if loss < bessloss:
            with open("./result/" + output_file, "a") as f:
                print(f'New best model found at epoch {epoch} with loss {loss:.4f} (previous best {bessloss:.4f})', file=f)
                # print(f'State at this epoch: {state}', file=f)
            bessloss = loss
            Z_final=Z_current
            pred_final = pred 
            ari_final = ari_score
            nmi_final = nmi_score 
            sil_final = silh_score
            # best_model_state = copy.deepcopy(scAGCLmodel.state_dict())
            best_model_state = state

        # --- Logic Early Stopping và Lưu trữ (mỗi epoch) ---

        # if epoch > 200: 
        #     sil_res.append(silh_score)
        #     sil_array = np.array(sil_res)

            # if silh_score > sil_max:
            #     Z_final = Z_current 
            #     pred_final = pred
            #     sil_max = silh_score
            #     ari_final = ari_score
            #     nmi_final = nmi_score 
            #     best_model_state = state

            # if len(sil_array) >= 100: 
            #     mean_0_n = np.mean(sil_array[-50:])
            #     mean_n_2n = np.mean(sil_array[-100:-50])

            #     if mean_0_n - mean_n_2n <= 0.01: 
            #         Z_final = Z_current
            #         ari_final = ari_score
            #         nmi_final = nmi_score 
            #         print('Stop early at', epoch, 'epoch')
            #         # with open("./result/" + output_file, "a") as f:
            #         #     print("Stop early: " + f'{loss.item():.4f}' + '\t' + 'ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score) + ', sil=' + str(silh_score), file=f)
            #         break

    # print('Pretrain result: ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', bessloss=' + str(bessloss))
    print('Pretrain result: ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil score=' + str(sil_final))

    # with open("./result/" + output_file, "a") as f:
    #     print("Pretrain result  is: " + 'ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final)+ ', sil score=' + str(sil_max), file=f)    
    
    with open("./result/" + output_file, "a") as f:
        print("Pretrain result  is: " + 'ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final)+ ', sil score=' + str(sil_final), file=f)

    return Z_final, pred_final, ari_final, nmi_final, best_model_state

# def Train(data_file, scAGCLmodel, data, cellGgraph, device, num_epochs, lam, num_cluster, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components):
#     #For visualize with tensorboard
#     # writer = SummaryWriter(log_dir="/content/drive/MyDrive/ĐACN-ĐATN/scAGCL/visualize_tensorboard")
#     start_time = t()
    
#     Z_final=None
#     bessloss = float('inf') # Khởi tạo bessloss là vô cùng lớn

#     ari_final, nmi_final = 0, 0 
    
#     # Lấy X và Adj full-graph một lần bên ngoài vòng lặp
#     X=data.x
#     full_edgeind = data.edge_index.to(device)
#     full_edgeind_self_loops, _ = add_self_loops(full_edgeind, num_nodes = X.shape[0])
#     Adj = to_dense_adj(full_edgeind_self_loops, max_num_nodes= X.shape[0])[0]
#     Adj = torch.clamp(Adj, 0, 1)
#     X = X.to(device)
#     Adj = Adj.to(device)
    
#     for epoch in range(1, num_epochs + 1):
#         scAGCLmodel.train() 
        
#         # --- Lấy Subgraph ---
#         subGraph = cellGgraph.subgraph(np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size])
#         x_sub = data.x[np.array(subGraph.nodes())].to(device)
#         subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
#         edgeind = np.array(subGraph .edges()).T
#         edgeind = torch.from_numpy(edgeind).to(device).long()
#         edgeind_self_loops, _ = add_self_loops(edgeind, num_nodes = x_sub.shape[0])
#         adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
#         adj = torch.clamp(adj, 0, 1)

#         # --- Tính Loss Tương phản (LossA) ---
#         optimizer.zero_grad()
#         x_1 = GeneDropping(x_sub, feature_r1)
#         x_2 = GeneDropping(x_sub, feature_r2)

#         edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
#         edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
#         edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
#         edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])

#         adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
#         adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
#         adj_1 = torch.clamp(adj_1, 0,1)
#         adj_2 = torch.clamp(adj_2, 0,1)

#         z_1 = scAGCLmodel(x_1, adj_1)
#         z_2 = scAGCLmodel(x_2, adj_2)    
#         loss1= scAGCLmodel.loss(z_1,z_2,batch_size=0)
        
#         loss2=0
#         if lam > 0:
#             adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
#             adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
#             z_3 = scAGCLmodel(x_3,adj_3)
#             z_4 = scAGCLmodel(x_4,adj_4)
#             loss2 = scAGCLmodel.loss(z_3,z_4,batch_size=0)
        
#         loss = loss1 + lam*loss2

#         # --- Backward và Step ---
#         # with torch.autograd.detect_anomaly(): 
#         loss.backward()
#         optimizer.step()
#         # (Kết thúc phần tính toán có gradient. Trọng số giờ là của Epoch N)
        
#         scAGCLmodel.eval() # Chuyển sang Eval mode
#         with torch.no_grad(): # TẮT GRADIENT
#             Z_current = scAGCLmodel(X, Adj) 
            
#             # --- Logging (Tính toán chỉ số) ---
#             Z_eval = Z_current.clone()
#             Y=data.y
#             Z = Z_eval.detach().cpu().numpy()
#             Y = Y.detach().cpu().numpy()
#             Z = normalize(Z, norm='l2')
#             kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=0)
#             pred = kmeans.fit_predict(Z)

#             ari_score = adjusted_rand_score(Y, pred)
#             nmi_score = normalized_mutual_info_score(Y, pred)

#             # --- In log (như cũ) ---
#             if epoch <= num_epochs and epoch%20==0:
#                 now_time = t()
#                 print(f'Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}, total loss={loss:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total time {now_time - start_time:.4f}')

#             # --- Lưu trữ mô hình tốt nhất ---
#             # Dùng loss.item() (loss của subgraph) để quyết định
#             current_loss = loss.item()
#             if current_loss < bessloss:
#                 bessloss = current_loss
#                 Z_final = Z_current.clone() # Lưu Z_current (full-graph) tương ứng
#                 ari_final = ari_score
#                 nmi_final = nmi_score 
    
#     # (Kết thúc vòng lặp for)

#     print('Pretrain result (best loss): ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final))

#     # with open("./result/" + output_file, "a") as f:
#     #     print("Pretrain result (best loss) is: " + 'ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final), file=f)    
    
#     # Trả về Z_final (là Z_current của epoch có loss tốt nhất)
#     return Z_final


def Finetuning(finetune_model, Z_pretrain, sil_max, ari, nmi, data, cellGgraph, device, num_epochs, lam, gam, num_cluster, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components):
    print(f'Finetuing with gam = {gam} and finetune_lr = {optimizer.param_groups[0]["lr"]} ...')
    with open("./result/" + output_file, "a") as f:
        print(f'Finetuing with gam = {gam} and finetune_lr = {optimizer.param_groups[0]["lr"]} ...', file=f)
    
    finetune_model.to(device)

    # 1. Khởi tạo tâm cụm
    cluster_centers = InitClusterCenters(embedding=Z_pretrain, num_cluster=num_cluster, device=device)
    # # Gán vào buffer của model
    finetune_model.cluster_centers.data.copy_(cluster_centers)
    
    start_time = t()
    
    ari_final, nmi_final = ari, nmi
    sil_res = []
    
    pred_final = None
    best_loss = None 

    # 2. Lấy X và Adj full-graph
    X=data.x
    full_edgeind = data.edge_index.to(device)
    full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
    full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0])
    # full_edgeind_self_loops = full_edgeind_symmetric
    Adj = to_dense_adj(full_edgeind_self_loops, max_num_nodes= X.shape[0])[0]
    Adj = torch.clamp(Adj, 0, 1)
    X = X.to(device)
    Adj = Adj.to(device)
    # full_edgeind_self_loops.to(device)

    # 3. Khởi tạo p_target_full lần đầu tiên
    print("Initializing target distribution P (scGAC method)...")
    finetune_model.eval()
    with torch.no_grad():
        Z_init = finetune_model(X, full_edgeind_self_loops, None)
        # Z_init = finetune_model(X, Adj)
        q_init = finetune_model.calculate_q(Z_init)
        p_target_full = finetune_model.calculate_p(q_init)
        
    update_interval = 1

    for epoch in range(1, num_epochs + 1): 
        # ==================================================================
        # PHẦN 1: LOGGING VÀ CẬP NHẬT P/TÂM CỤM (Eval Mode)
        # ==================================================================
        finetune_model.eval() 
        with torch.no_grad(): 
            
            # Z_current = finetune_model(X, full_edgeind_self_loops, None) 
            Z_current = finetune_model(X, Adj)
            
            # --- Logging (Tính toán chỉ số) ---
            Z_eval = Z_current.clone()
            Y=data.y
            Z_raw = Z_eval.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            Z_norm = normalize(Z_raw, norm='l2')
            kmeans = KMeans(n_clusters=num_cluster, init="k-means++", n_init=20, random_state=0)
            pred = kmeans.fit_predict(Z_norm)

            ari_score = adjusted_rand_score(Y, pred)
            nmi_score = normalized_mutual_info_score(Y, pred)
            
            if len(np.unique(pred)) > 1:
                silh_score = silhouette_score(Z_norm, pred)
            else:
                silh_score = -1.0 
            
            # --- Cập nhật P VÀ TÂM CỤM (Nếu đến kỳ) ---
            if epoch % update_interval == 0:
                print(f"Epoch {epoch}: Updating target P and Centers (scGAC method)...")
                q_full = finetune_model.calculate_q(Z_current)
                p_target_full = finetune_model.calculate_p(q_full) 

                # print(f'Cluster centers at epoch {epoch:03d}: {finetune_model.cluster_centers.data}')
                
                # CẬP NHẬT TÂM CỤM THỦ CÔNG (Phương trình 10)
                finetune_model.update_clusters_center(Z_current, q_full, num_cluster, device)


        finetune_model.train() 
        
        # --- Lấy Subgraph ---
        subgraph_nodes_indices = np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size]
        subGraph = cellGgraph.subgraph(subgraph_nodes_indices)
        x_sub = data.x[np.array(subGraph.nodes())].to(device)
        subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
        edgeind = np.array(subGraph .edges()).T
        edgeind = torch.from_numpy(edgeind).to(device).long()
        edgeind_symmetric = torch.cat([edgeind, edgeind.flip(0)], dim=1)
        # edgeind_self_loops = edgeind_symmetric
        edgeind_self_loops, _ = add_self_loops(edgeind_symmetric, num_nodes = x_sub.shape[0])
        adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
        adj = torch.clamp(adj, 0, 1)

        # --- Tính Loss Tương phản (LossA) ---
        optimizer.zero_grad()
        x_1 = GeneDropping(x_sub, feature_r1)
        x_2 = GeneDropping(x_sub, feature_r2)

        edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
        edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
        edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
        edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])
        # edgeind_1_self_loops = edgeind_1
        # edgeind_2_self_loops = edgeind_2

        adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
        adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
        adj_1 = torch.clamp(adj_1, 0,1)
        adj_2 = torch.clamp(adj_2, 0,1)

        z_1 = finetune_model(x_1, adj_1)
        z_2 = finetune_model(x_2, adj_2)   
        # z_1 = finetune_model(x_1, edgeind_1_self_loops, None)
        # z_2 = finetune_model(x_2, edgeind_2_self_loops, None)
        loss1= finetune_model.loss(z_1,z_2,batch_size=0)
        
        loss2=0
        if lam > 0:
            adj_3, x_3 = GraphAdversarialAttack(finetune_model, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
            adj_4, x_4 = GraphAdversarialAttack(finetune_model, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
            # adj_3, x_3 = GraphAdversarialAttack(finetune_model, edgeind_self_loops, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
            # adj_4, x_4 = GraphAdversarialAttack(finetune_model, edgeind_self_loops, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
            # edgeind_3, edge_weight_3 = dense_to_differentiable_sparse(adj_3)
            # z_3 = finetune_model(x_3, edgeind_3, edge_weight_3)
            # z_3 = scAGCLmodel(x_3, adj_3, None)
            # z_3 = scAGCLmodel(x_3, adj_3)
            # edgeind_4, edge_weight_4 = dense_to_differentiable_sparse(adj_4)
            # z_4 = finetune_model(x_4, edgeind_4, edge_weight_4)
            z_3 = finetune_model(x_3,adj_3)
            z_4 = finetune_model(x_4,adj_4)
            loss2 = finetune_model.loss(z_3,z_4,batch_size=0)
        
        lossA = loss1 + lam*loss2
        
        # --- Tính Loss Phân cụm (c_loss) ---
        # Tâm cụm (self.cluster_centers) được tính từ full-graph ở epoch trước
        # Z_sub = finetune_model(x_sub, adj) 
        # q_sub = finetune_model.calculate_q(Z_sub) 

        # Z_full = finetune_model(X, full_edgeind_self_loops, None)
        Z_full = finetune_model(X, Adj)
        q_full = finetune_model.calculate_q(Z_full)
        
        # p_target_sub = p_target_full[subgraph_nodes_indices]
        
        # c_loss = finetune_model.clustering_loss_new1(q_sub, p_target_sub)

        c_loss = finetune_model.clustering_loss_new1(q_full, p_target_full)

        # --- Backward và Step ---
        loss = lossA + gam * c_loss 
        # loss = loss1 + gam * c_loss

        # loss = gam * c_loss

        # loss = lossA
        
        # with torch.autograd.detect_anomaly(): 
        loss.backward()
          
        # optimizer.step() sẽ CHỈ cập nhật GAT ENCODER
        optimizer.step()
        
        
            
        # --- In log (mỗi 5 epoch) ---
        if epoch % 1 == 0:
                now_time = t()
                # print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
                print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
                # print(f'Epoch={epoch:03d}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
                # print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')

            # --- Logic Early Stopping và Lưu trữ (mỗi epoch) ---
        sil_res.append(silh_score)
        sil_array = np.array(sil_res)

        if silh_score >= sil_max:
            # if epoch == 1:
            #     # best_loss = loss
                sil_max = silh_score
            #     pred_final = pred

            # if loss < best_loss:
                # best_loss = loss
                pred_final = pred 
                ari_final = ari_score
                nmi_final = nmi_score 
                # sil_final = silh_score

        if len(sil_array) >= 100: 
                mean_0_n = np.mean(sil_array[-50:])
                mean_n_2n = np.mean(sil_array[-100:-50])

                if mean_0_n - mean_n_2n <= 0.01: 
                    print('Stop early at', epoch, 'epoch')
                    # with open("./result/" + output_file, "a") as f:
                    #     print("Stop early: " + f'{loss.item():.4f}' + '\t' + 'ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score) + ', sil=' + str(silh_score), file=f)
                    break
            
    # (Kết thúc vòng lặp for)

    # In kết quả cuối cùng (dựa trên min loss)
    print('Final Finetune result (best silhouette): ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score=' + str(sil_max) + ', min loss=' + str(best_loss))
    with open("./result/" + output_file, "a") as f:
        print('Final Finetune result is: ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score=' + str(sil_max) + ', min loss=' + str(best_loss), file=f)    
    
    with open("./result/total.csv", "a") as tf:
        print(output_file + " (Finetune): " + 'ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score' + str(sil_max) + ', min loss=' + str(best_loss), file=tf)


# def Finetuning(finetune_model, Z_pretrain, sil_max, ari, nmi, data, cellGgraph, device, num_epochs, lam, gam, num_cluster, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components):
#     print(f'START SPECTRAL FINETUNING with gam = {gam} ...')
#     finetune_model.to(device)
    
#     # Lưu ý: Spectral Clustering không cần KMeans init tâm cụm
#     # Nó học trực tiếp ma trận Assignment Y thông qua loss graph cut
#     Z_np = normalize(Z_pretrain.cpu().detach().numpy(), norm='l2') # Cosine metric
    
#     # 2. Chạy KMeans để tìm tâm cụm khởi đầu
#     kmeans = KMeans(n_clusters=num_cluster, init="k-means++", n_init=20, random_state=0)
#     kmeans.fit(Z_np)
#     centers = kmeans.cluster_centers_ # Shape: (K, Hidden)
    
#     # 3. Gán tâm cụm vào trọng số của cluster_layer
#     # Linear layer: y = xW^T + b. Weight shape là (Out, In) tức (K, Hidden)
#     # Ta muốn activation lớn nhất khi x giống center nhất (dot product)
#     with torch.no_grad():
#         # Chuyển center thành tensor
#         centers_tensor = torch.from_numpy(centers).float().to(device)
        
#         # Gán weight. Lưu ý: cluster_layer được định nghĩa trong model.py
#         # Nếu đệ dùng nn.Linear, weight shape là (num_clusters, num_hidden)
#         finetune_model.cluster_layer.weight.data.copy_(centers_tensor)
        
#         # Gán bias bằng 0 (để thuần túy là dot product similarity)
#         if finetune_model.cluster_layer.bias is not None:
#             finetune_model.cluster_layer.bias.data.fill_(0)
#     start_time = t()
#     ari_final, nmi_final = ari, nmi
    
#     # Chuẩn bị dữ liệu Full Graph (cho Spectral Loss)
#     X = data.x.to(device)
#     full_edgeind = data.edge_index.to(device)
#     # Tạo symmetric edges (Vô hướng) cho Spectral Clustering
#     full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
#     # Self-loops sẽ được thêm bên trong hàm spectral_loss_sparse
#     full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0])
#     full_edgeind_self_loops.to(device)
    
#     # List lưu lịch sử ARI để chọn model tốt nhất (vì Silhoutte không còn đáng tin cậy tuyệt đối)
#     best_ari = 0.0
#     sil_res = []
    
#     for epoch in range(1, num_epochs + 1): 
#         finetune_model.train() 
#         # --- Lấy Subgraph ---
#         subgraph_nodes_indices = np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size]
#         subGraph = cellGgraph.subgraph(subgraph_nodes_indices)
#         x_sub = data.x[np.array(subGraph.nodes())].to(device)
#         subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
#         edgeind = np.array(subGraph .edges()).T
#         edgeind = torch.from_numpy(edgeind).to(device).long()
#         edgeind_self_loops, _ = add_self_loops(edgeind, num_nodes = x_sub.shape[0])
#         adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
#         adj = torch.clamp(adj, 0, 1)

#         # --- Tính Loss Tương phản (LossA) ---
#         optimizer.zero_grad()
#         x_1 = GeneDropping(x_sub, feature_r1)
#         x_2 = GeneDropping(x_sub, feature_r2)

#         edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
#         edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
#         edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
#         edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])

#         adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
#         adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
#         adj_1 = torch.clamp(adj_1, 0,1)
#         adj_2 = torch.clamp(adj_2, 0,1)

#         z_1 = finetune_model(x_1, adj_1)
#         z_2 = finetune_model(x_2, adj_2)    
#         loss1= finetune_model.loss(z_1,z_2,batch_size=0)
        
#         loss2=0
#         if lam > 0:
#             adj_3, x_3 = GraphAdversarialAttack(finetune_model, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
#             adj_4, x_4 = GraphAdversarialAttack(finetune_model, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
#             z_3 = finetune_model(x_3,adj_3)
#             z_4 = finetune_model(x_4,adj_4)
#             loss2 = finetune_model.loss(z_3,z_4,batch_size=0)
        
#         lossA = loss1 + lam*loss2
#         # optimizer.zero_grad()
        
#         # 1. Forward Pass trên Full Graph để lấy Y (Assignment) và Z
#         # Spectral Loss cần cái nhìn toàn cục (Global View)
#         Y_soft, Z_full = finetune_model.forward_cluster(X, full_edgeind_self_loops)
        
#         # 2. Tính Spectral Loss (MinCut + Orthogonality)
#         # Dùng trực tiếp edge_index (Sparse) để tiết kiệm bộ nhớ
#         spec_loss = finetune_model.spectral_loss_sparse(Y_soft, full_edgeind_self_loops)
        
#         # 3. Tính Contrastive Loss (Optional nhưng khuyến nghị)
#         # Giúp duy trì đặc trưng tốt từ Pretrain, tránh bị biến dạng quá nhiều
#         # Ta có thể lấy subgraph để tính contrastive loss cho nhẹ (như cũ)
#         # Hoặc tạm thời tắt nó đi (lossA = 0) để kiểm tra sức mạnh của Spectral thuần túy.
#         # Ở đây đại ca để lossA = 0 để đệ test Spectral trước. 
#         # Nếu muốn, đệ uncomment đoạn dưới để thêm augmentation loss.
#         # lossA = torch.tensor(0.0, device=device)
        
#         # --- Code tính LossA (nếu cần) ---
#         # ... (Lấy subgraph, augment, tính loss1, loss2 như pretrain) ...
#         # ---------------------------------

#         # Tổng loss
#         loss = lossA + gam * spec_loss
        
#         loss.backward()
#         optimizer.step()
        
#         # --- Đánh giá (Eval) ---
#         if epoch % 1 == 0:
#             finetune_model.eval() 
#             with torch.no_grad():
#                 Y_eval, Z_eval = finetune_model.forward_cluster(X, full_edgeind_symmetric)
#                 pred = torch.argmax(Y_eval, dim=1).cpu().numpy()
                
#                 # Ground Truth (chỉ để in log, không dùng để stop)
#                 Y_true = data.y.cpu().numpy()
                
#                 # Tính Metrics
#                 ari_score = adjusted_rand_score(Y_true, pred)
#                 nmi_score = normalized_mutual_info_score(Y_true, pred)
                
#                 # Tính Silhouette (Dùng để Early Stopping)
#                 # Dùng Z_eval (Embedding) để tính độ gọn
#                 Z_np = Z_eval.detach().cpu().numpy()
#                 # Normalize Z trước khi tính Sil (thường tốt hơn cho metric cosine-based)
#                 Z_np = normalize(Z_np, norm='l2') 
                
#                 if len(np.unique(pred)) > 1:
#                     silh_score = silhouette_score(Z_np, pred)
#                 else:
#                     silh_score = -1.0

#                 now_time = t()
#                 print(f'Epoch={epoch:03d}, Loss={loss.item():.4f} LossA={lossA.item():.4f} Spec={spec_loss.item():.4f}, Sil={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, Time={now_time - start_time:.2f}')
                
#                 # --- Logic Early Stopping và Lưu trữ (mỗi epoch) ---
#                 sil_res.append(silh_score)
#                 sil_array = np.array(sil_res)

#                 if silh_score >= sil_max:
#                     pred_final = pred
#                     sil_max = silh_score
#                     ari_final = ari_score
#                     nmi_final = nmi_score 

#                 if len(sil_array) >= 100: 
#                     mean_0_n = np.mean(sil_array[-50:])
#                     mean_n_2n = np.mean(sil_array[-100:-50])

#                     if mean_0_n - mean_n_2n <= 0.02: 
#                         print('Stop early at', epoch, 'epoch')
#                         # with open("./result/" + output_file, "a") as f:
#                         #     print("Stop early: " + f'{loss.item():.4f}' + '\t' + 'ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score) + ', sil=' + str(silh_score), file=f)
#                         break

#     print(f'Final Spectral Result: ARI={ari_final:.4f}, NMI={nmi_final:.4f}')
    
#     with open("./result/" + output_file, "a") as f:
#         print(f'Spectral Finetune: ARI={ari_final:.4f}, NMI={nmi_final:.4f}', file=f)


# def Finetuning(finetune_model, Z_pretrain, sil_max, ari, nmi, data, cellGgraph, device, num_epochs, lam, gam, num_cluster, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size, output_file, principal_components):
#     print(f'PCL Finetuing with gam = {gam} and finetune_lr = {optimizer.param_groups[0]["lr"]} ...')
#     with open("./result/" + output_file, "a") as f:
#         print(f'Finetuing with gam = {gam} and finetune_lr = {optimizer.param_groups[0]["lr"]} ...', file=f)
    
#     finetune_model.to(device)

#     # 1. Khởi tạo tâm cụm
#     # cluster_centers = InitClusterCenters(embedding=Z_pretrain, num_cluster=num_cluster, device=device)
#     # # Gán vào buffer của model
#     # finetune_model.cluster_centers.data.copy_(cluster_centers)
    
#     start_time = t()
    
#     ari_final, nmi_final = ari, nmi
#     sil_res = []
    
#     pred_final = None
#     best_loss = None 

#     # 2. Lấy X và Adj full-graph
#     X=data.x
#     full_edgeind = data.edge_index.to(device)
#     full_edgeind_symmetric = torch.cat([full_edgeind, full_edgeind.flip(0)], dim=1)
#     full_edgeind_self_loops, _ = add_self_loops(full_edgeind_symmetric, num_nodes = X.shape[0])
#     # Adj = to_dense_adj(full_edgeind_self_loops, max_num_nodes= X.shape[0])[0]
#     # Adj = torch.clamp(Adj, 0, 1)
#     X = X.to(device)
#     # Adj = Adj.to(device)
#     full_edgeind_self_loops.to(device)

#     # 3. Khởi tạo p_target_full lần đầu tiên
#     # print("Initializing target distribution P (scGAC method)...")
#     # finetune_model.eval()
#     # with torch.no_grad():
#     #     Z_init = finetune_model(X, full_edgeind_self_loops)
#     #     # Z_init = finetune_model(X, Adj)
#     #     q_init = finetune_model.calculate_q(Z_init)
#     #     p_target_full = finetune_model.calculate_p(q_init)
        
#     update_interval = 1

#     pseudo_labels = torch.zeros(X.shape[0], dtype=torch.long).to(device)

#     for epoch in range(1, num_epochs + 1): 
#         # ==================================================================
#         # PHẦN 1: LOGGING VÀ CẬP NHẬT P/TÂM CỤM (Eval Mode)
#         # ==================================================================
#         finetune_model.eval() 
#         with torch.no_grad(): 
            
#             Z_current = finetune_model(X, full_edgeind_self_loops) 
#             # Z_current = finetune_model(X, Adj)
            
#             # --- Logging (Tính toán chỉ số) ---
#             Z_eval = Z_current.clone()
#             Y=data.y
#             Z_raw = Z_eval.detach().cpu().numpy()
#             Y = Y.detach().cpu().numpy()
#             Z_norm = normalize(Z_raw, norm='l2')
#             kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=0)
#             pred = kmeans.fit_predict(Z_norm)

#             ari_score = adjusted_rand_score(Y, pred)
#             nmi_score = normalized_mutual_info_score(Y, pred)
            
#             if len(np.unique(pred)) > 1:
#                 silh_score = silhouette_score(Z_norm, pred)
#             else:
#                 silh_score = -1.0 
            
#             # --- Cập nhật P VÀ TÂM CỤM (Nếu đến kỳ) ---
#             if epoch % update_interval == 0:
#                 # print(f"Epoch {epoch}: Updating target P and Centers (scGAC method)...")
#                 # q_full = finetune_model.calculate_q(Z_current)
#                 # p_target_full = finetune_model.calculate_p(q_full) 

#                 # print(f'Cluster centers at epoch {epoch:03d}: {finetune_model.cluster_centers.data}')
                
#                 # CẬP NHẬT TÂM CỤM THỦ CÔNG (Phương trình 10)
#                 # finetune_model.update_clusters_center(Z_current, q_full, num_cluster, device)

#                 centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
#                 centroids = F.normalize(centroids, p=2, dim=1)
#                 finetune_model.prototypes.data.copy_(centroids)

#                 pseudo_labels = torch.tensor(pred, dtype=torch.long).to(device)

        
#         finetune_model.train() 
        
#         # --- Lấy Subgraph ---
#         subgraph_nodes_indices = np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size]
#         subGraph = cellGgraph.subgraph(subgraph_nodes_indices)
#         x_sub = data.x[np.array(subGraph.nodes())].to(device)
#         subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')     

#         sub_pseudo_labels = pseudo_labels[subgraph_nodes_indices]

#         edgeind = np.array(subGraph .edges()).T
#         edgeind = torch.from_numpy(edgeind).to(device).long()
#         edgeind_self_loops, _ = add_self_loops(edgeind, num_nodes = x_sub.shape[0])
#         adj = to_dense_adj(edgeind_self_loops, max_num_nodes=x_sub.shape[0])[0]
#         adj = torch.clamp(adj, 0, 1)

#         # --- Tính Loss Tương phản (LossA) ---
#         optimizer.zero_grad()
#         x_1 = GeneDropping(x_sub, feature_r1)
#         x_2 = GeneDropping(x_sub, feature_r2)

#         edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
#         edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
#         edgeind_1_self_loops, _ = add_self_loops(edgeind_1, num_nodes=x_1.shape[0])
#         edgeind_2_self_loops, _ = add_self_loops(edgeind_2, num_nodes=x_2.shape[0])

#         adj_1 = to_dense_adj(edgeind_1_self_loops, max_num_nodes=x_1.shape[0])[0]
#         adj_2 = to_dense_adj(edgeind_2_self_loops, max_num_nodes=x_2.shape[0])[0]
#         adj_1 = torch.clamp(adj_1, 0,1)
#         adj_2 = torch.clamp(adj_2, 0,1)

#         z_1 = finetune_model(x_1, adj_1)
#         z_2 = finetune_model(x_2, adj_2)    
#         loss1= finetune_model.loss(z_1,z_2,batch_size=0)

#         loss_proto_1 = finetune_model.proto_contrastive_loss(z_1, sub_pseudo_labels)
#         loss_proto_2 = finetune_model.proto_contrastive_loss(z_2, sub_pseudo_labels)
#         loss1_proto = (loss_proto_1 + loss_proto_2) / 2.0
        
#         loss2=0
#         loss2_proto = 0
#         if lam > 0:
#             adj_3, x_3 = GraphAdversarialAttack(finetune_model, adj, adj_1, x_sub, x_1, iters, 0.2, alpha, beta, principal_components)
#             adj_4, x_4 = GraphAdversarialAttack(finetune_model, adj, adj_2, x_sub, x_2, iters, 0.2, alpha, beta, principal_components)
#             z_3 = finetune_model(x_3,adj_3)
#             z_4 = finetune_model(x_4,adj_4)
#             loss2 = finetune_model.loss(z_3,z_4,batch_size=0)

#             loss_proto_3 = finetune_model.proto_contrastive_loss(z_3, sub_pseudo_labels)
#             loss_proto_4 = finetune_model.proto_contrastive_loss(z_4, sub_pseudo_labels)
#             loss2_proto = (loss_proto_3 + loss_proto_4) / 2.0
        
#         lossA = loss1 + lam*loss2
        
#         # --- Tính Loss Phân cụm (c_loss) ---
#         # Tâm cụm (self.cluster_centers) được tính từ full-graph ở epoch trước
#         # Z_sub = finetune_model(x_sub, adj) 
#         # q_sub = finetune_model.calculate_q(Z_sub) 

#         # Z_full = finetune_model(X, full_edgeind_self_loops)
#         # q_full = finetune_model.calculate_q(Z_full)
        
#         # p_target_sub = p_target_full[subgraph_nodes_indices]
        
#         # c_loss = finetune_model.clustering_loss_new1(q_sub, p_target_sub)

#         # c_loss = finetune_model.clustering_loss_new1(q_full, p_target_full)

#         # --- Backward và Step ---
#         # loss = lossA + gam * c_loss 

#         loss_proto = loss1_proto + lam * loss2_proto

#         loss = lossA + gam * loss_proto
#         # loss = loss1 + lam*loss2 + gam * (loss1_proto + lam * loss2_proto)
#         # loss = loss1 + gam * loss1_proto + lam * (loss2 + gam * loss2_proto)

#         # loss = loss1 + gam * c_loss

#         # loss = gam * c_loss

#         # loss = lossA
        
#         # with torch.autograd.detect_anomaly(): 
#         loss.backward()
          
#         # optimizer.step() sẽ CHỈ cập nhật GAT ENCODER
#         optimizer.step()
        
#         # --- In log (mỗi 5 epoch) ---
#         if epoch % 1 == 0:
#             now_time = t()
#                 # print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
#                 # print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
#                 # print(f'Epoch={epoch:03d}, c_loss={c_loss.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
#                 # print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')
#             print(f'Epoch={epoch:03d}, lossA={lossA.item():.4f}, loss_proto={loss_proto.item():.4f}, sil_score={silh_score:.4f}, ARI={ari_score:.4f}, NMI={nmi_score:.4f}, total loss={loss.item():.4f}, total time {now_time - start_time:.4f}')

#         # --- Logic Early Stopping và Lưu trữ (mỗi epoch) ---
#         sil_res.append(silh_score)
#         sil_array = np.array(sil_res)

#         if silh_score >= sil_max:
#         # if epoch == 1:
#         #     # best_loss = loss
#             sil_max = silh_score
#         #     pred_final = pred

#         # if loss < best_loss:
#             # best_loss = loss
#             pred_final = pred 
#             ari_final = ari_score
#             nmi_final = nmi_score 
#             # sil_final = silh_score

#         if len(sil_array) >= 100: 
#             mean_0_n = np.mean(sil_array[-50:])
#             mean_n_2n = np.mean(sil_array[-100:-50])

#             if mean_0_n - mean_n_2n <= 0.01: 
#                 print('Stop early at', epoch, 'epoch')
#                 # with open("./result/" + output_file, "a") as f:
#                 #     print("Stop early: " + f'{loss.item():.4f}' + '\t' + 'ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score) + ', sil=' + str(silh_score), file=f)
#                 break
            
#     # (Kết thúc vòng lặp for)

#     # In kết quả cuối cùng (dựa trên min loss)
#     print('Final Finetune result (best silhouette): ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score=' + str(sil_max) + ', min loss=' + str(best_loss))
#     with open("./result/" + output_file, "a") as f:
#         print('Final Finetune result is: ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score=' + str(sil_max) + ', min loss=' + str(best_loss), file=f)    
    
#     with open("./result/total.csv", "a") as tf:
#         print(output_file + " (Finetune): " + 'ARI= ' + str(ari_final) + ', NMI=' + str(nmi_final) + ', sil_score' + str(sil_max) + ', min loss=' + str(best_loss), file=tf)