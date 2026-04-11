# This file is implemented based on this repository: https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce, softmax, contains_self_loops, to_dense_adj
# from sparsemax import Sparsemax

# torch.set_printoptions(profile="full")


class GATLayerHybrid(nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True):
        
        super().__init__()
        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        # if add_skip_connection:
        #     self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        # else:
        #     self.register_parameter('skip_proj', None)

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, graph, edge_weight=None):
        # x shape = (N, Fin), graph shape = (N, N) or (2, E)
        num_nodes = x.shape[0]

        # x = self.dropout(x)

        # (N, Fin) -> (N, NH, Fout)
        nodes_features_proj = self.linear_proj(x).view(num_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        if graph.is_sparse or (graph.dim() == 2 and graph.shape[0] == 2):
            edge_index = graph
            if edge_index.is_sparse:
                edge_index = edge_index._indices()

            scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

            src_nodes_index = edge_index[self.src_nodes_dim]
            trg_nodes_index = edge_index[self.trg_nodes_dim]
            scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
            scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)

            # (E, NH)
            scores_per_edge = self.leaky_ReLU(scores_source_lifted + scores_target_lifted)

            # --- Sparse Softmax ---
            # scores_per_edge shape: [E, NH]
            # trg_nodes_index shape: [E]
            # out shape: [N, NH]
            indices_transposed = torch.stack([trg_nodes_index, src_nodes_index])
            sparse_logits = torch.sparse_coo_tensor(
                indices_transposed,
                scores_per_edge, 
                (num_nodes, num_nodes, self.num_of_heads)
            )
            sparse_attention = torch.sparse.softmax(sparse_logits, dim=1)
            attention_per_edge = sparse_attention.values() # (E, NH)
            attention_indices = sparse_attention.indices()
            attention_per_edge = self.dropout(attention_per_edge)
            
            outputs_per_head = []
            for head in range(self.num_of_heads):
                alpha_h = attention_per_edge[:, head]
                
                # Shape (N, N)
                adj_sparse_h = torch.sparse_coo_tensor(
                    attention_indices, 
                    alpha_h, 
                    (num_nodes, num_nodes)
                )
                adj_sparse_h = adj_sparse_h.coalesce()
                h_prime = nodes_features_proj[:, head, :]
                
                # (N, N) x (N, Fout) -> (N, Fout)
                out_h = torch.sparse.mm(adj_sparse_h, h_prime)
                outputs_per_head.append(out_h)
            
            # (N, NH, Fout)
            out_nodes_features = torch.stack(outputs_per_head, dim=1)

        elif graph.dim() == 2 and graph.shape[0] == graph.shape[1]:
            adj = graph # (N, N)

            scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
            scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

            scores_target = scores_target.transpose(0, 1)
            scores_source = scores_source.permute(1, 2, 0)
            
            # (N, N, NH)
            all_scores = self.leaky_ReLU(scores_target + scores_source)

            # Masking
            additive_mask = (1.0 - adj) * -1e9
            attention_logits = all_scores + additive_mask.unsqueeze(0)
            
            # Softmax
            all_attention_coefficients = F.softmax(attention_logits, dim=-1) # (N, N, NH)

            all_attention_coefficients = self.dropout(all_attention_coefficients)

            # Aggregate
            # (NH, N, N) x (NH, N, Fout) -> (NH, N, Fout)
            out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
            out_nodes_features = out_nodes_features.permute(1, 0, 2)
            
        else:
            raise ValueError("Đầu vào không xác định: không phải Adj (N,N) hay edge_index (2,E).")

        if not out_nodes_features.is_contiguous():
            # print("Warning: out_nodes_features is not contiguous. Making it contiguous now.")
            out_nodes_features = out_nodes_features.contiguous()

        if self.concat:
            out_nodes_features = out_nodes_features.reshape(num_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)

        return out_nodes_features

class GATEncoderHybrid(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob=0.6):
        super().__init__()

        self.gat_layer_1 = GATLayerHybrid(
            num_in_features= in_features,
            num_out_features= hidden_features,
            num_of_heads= num_heads,
            concat= True,
            activation= nn.ReLU(),
            # activation= nn.ELU(),
            dropout_prob= dropout_prob,
            add_skip_connection=False #
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.gat_layer_2 = GATLayerHybrid(
            num_in_features= hidden_features * num_heads,
            num_out_features= out_features,
            num_of_heads= 1,
            concat= False,
            activation= None,
            dropout_prob= dropout_prob,
            add_skip_connection=False
        )

    def forward(self, x, graph, edge_weight=None):
        x = self.gat_layer_1(x, graph, edge_weight)
        x = self.dropout(x)
        x = self.gat_layer_2(x, graph, edge_weight)
        return x