# This file is implemented based on this repository: https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
from utils_run import edge2adj, normalize_adj_tensor
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce, softmax, contains_self_loops, to_dense_adj
# from sparsemax import Sparsemax

# torch.set_printoptions(profile="full")

class GATLayer(nn.Module):
    src_nodes_dim = 0 # position of source nodes in edge index
    trg_nodes_dim = 1 # position of target node in edge index

    nodes_dim = 0 # node dimension/axis
    head_dim = 1 # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.6, add_skip_connection=True, bias=True):
        
        super().__init__()

        # self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat
        # self.activation = activation
        # self.dropout_prob = dropout_prob
        self.add_skip_connection = add_skip_connection
        # self.bias = bias

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # Linear projection matrix W
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)

        self.init_params()

    
    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    
    def forward(self, data):
        # Note that data here is a tuple: (in_nodes_features, edge_index)
        in_nodes_features, edge_index, edge_weight = data    # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=in_nodes_features.device)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_attr=edge_weight)
            
        # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=num_of_nodes)

        # edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_of_nodes, 'add')

        #
        # Step 1: Linear projection
        #

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        # nodes_features_proj = self.dropout(nodes_features_proj)     # in the official GAT imp they did dropout here as well


        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Note: Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)

        # Compute the attention coefficients for each edge
        
        # As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        # Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        # into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        # in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        # (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
        #  i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        # Note:
        # Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        # and it's a fairly common "trick" used in pretty much every deep learning framework.
        # Check out this link for more details:
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        scores_per_edge = self.leaky_ReLU(scores_source_lifted + scores_target_lifted)
        # print("scores_per_edge before masking:", scores_per_edge)

        # m = torch.rand_like(edge_weight)
        edge_weight_binary = (edge_weight > 0).float()
        edge_weight_differentiable = edge_weight_binary - edge_weight.detach() + edge_weight
        mask_value = (1.0 - edge_weight_differentiable.view(-1, 1)) * -1e9

        # mask_value = (1.0 - edge_weight.view(-1, 1)) * -100

        # mask_value = torch.log(edge_weight.view(-1, 1) + 1e-9)

        scores_per_edge = scores_per_edge + mask_value

        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        # Calculate the denominator. Shape = (E, NH)
        neighborhood_sum_denominator = self.sum_edge_scores_neighborhood(exp_scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attention_per_edge = exp_scores_per_edge / (neighborhood_sum_denominator + 1e-16)
        attention_per_edge = attention_per_edge.unsqueeze(-1)
        attention_per_edge = self.dropout(attention_per_edge)


        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attention_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)


        #
        # Step 4: Residual/skip connections, concat and bias
        #

        # if self.add_skip_connection: # add skip or residual connection
        #     if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: # if FIN == FOUT
        #         # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
        #         # thus we're basically copying input vectors NH times and adding to processed vectors
        #         out_nodes_features += in_nodes_features.unsqueeze(1)
        #     else:
        #         # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
        #         # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
        #         out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim = self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        # if (out_nodes_features < 0.0).any():
        #     print("Negative values in out_nodes_features")

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)

        return out_nodes_features, edge_index


    def sum_edge_scores_neighborhood(self, exp_scores_per_edge, trg_nodes_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_nodes_index_expanded = self.check_shape(trg_nodes_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sum = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sum.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sum.index_select(self.nodes_dim, trg_nodes_index)
    

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_nodes_index_expanded = self.check_shape(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)

        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, nodes_features_proj_lifted_weighted)

        return out_nodes_features


    def check_shape(self, this, other):
        # Append singleton dimension until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    

class GATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()

        # Layer 1: GAT with multi-head, output is concatenated and activated by ELU
        self.gat_layer_1 = GATLayer(
            num_in_features= in_features,
            num_out_features= hidden_features,
            num_of_heads= num_heads,
            concat= True,
            # activation= nn.ELU(),
            activation= nn.ReLU(),
            dropout_prob= dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)

        # Layer 2: GAT with multi-head, output is averaged and no activation
        self.gat_layer_2 = GATLayer(
            num_in_features= hidden_features * num_heads,
            num_out_features= out_features,
            num_of_heads= 1,
            concat= False,
            activation= None,
            dropout_prob= dropout_prob
        )

    
    def forward(self, x, edge_index, edge_weight=None):
        # if adj.is_sparse:
        #     edge_index = adj._indices()

        # else:
        #     edge_index = adj.nonzero().t().contiguous()
        # if isinstance(edge_index, tuple):
        #     edge_index = edge_index[0]
            
        x, _ = self.gat_layer_1((x, edge_index, edge_weight))
        x = self.dropout(x)
        x, _ = self.gat_layer_2((x, edge_index, edge_weight))

        return x

class GATLayerDense(nn.Module):
    """
    Phiên bản của lớp GAT có thể hoạt động với ma trận kề dày (dense).
    Điều này cho phép gradient chảy ngược qua ma trận kề,
    cần thiết cho các cuộc tấn công đối nghịch vào cấu trúc đồ thị.
    """
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=False, bias=True):
        super().__init__()

        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        # Ma trận trọng số W
        # self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # Vector chú ý a
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # import pdb; pdb.set_trace() 
        # x shape: (N, num_in_features), adj shape: (N, N)
        num_nodes = x.shape[0]

        if adj.is_sparse:
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj

        # --- CHÈN CODE KIỂM TRA VÀO ĐÂY ---
        # try:
        #     # 1. Tính tổng của mỗi hàng (dim=1)
        #     row_sums = dense_adj.sum(dim=1)
            
        #     # 2. Kiểm tra xem có hàng nào có tổng <= 0 không
        #     # (PGD có thể tạo ra số âm, nên ta kiểm tra <= 0 cho chắc)
        #     has_problematic_rows = torch.any(row_sums <= 0)
            
        #     if has_problematic_rows:
        #         problematic_indices = torch.where(row_sums <= 0)[0]
        #         print("="*50)
        #         print(f"!!! CẢNH BÁO TỪ GATConvDense !!!")
        #         print(f"Phát hiện {problematic_indices.numel()} hàng (node) có tổng <= 0 trong 'dense_adj'.")
        #         print(f"Đây là nguyên nhân GÂY RA NaN (log(0) hoặc 0/0).")
        #         print(f"Chỉ số của các hàng có vấn đề: {problematic_indices}")
        #         print("="*50)
        #         # Dừng chương trình để debug
        #         import pdb; pdb.set_trace() 
                
        # except Exception as e:
        #     print(f"Lỗi khi đang kiểm tra 'dense_adj': {e}")
        # --- KẾT THÚC CODE KIỂM TRA ---

        # Clamp value to 0-1
        adj_with_self_loops = dense_adj 
        # 1. Biến đổi đặc trưng tuyến tính
        nodes_features_proj = self.linear_proj(x).view(num_nodes, self.num_of_heads, self.num_out_features)
        if torch.isnan(nodes_features_proj).any(): 
          print("NaN in nodes_features_proj!")
        # nodes_features_proj = self.dropout(nodes_features_proj)
        # 2. Tính toán điểm chú ý cho tất cả các cặp node
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        if torch.isnan(scores_source).any(): 
          print("NaN in scores_source!")
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        if torch.isnan(scores_target).any(): 
          print("NaN in scores_target!")
        e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
        if torch.isnan(e).any(): 
          print("NaN in e!")

        # zero_vec = -1e9 * torch.ones_like(e)

        # additive_mask = (1.0 - adj_with_self_loops) * -1e5
        # attention_logits = e + additive_mask.unsqueeze(-1)

        attention_logits = e + torch.log(adj_with_self_loops.unsqueeze(-1) + 1e-9)

        # attention_logits = e * adj_with_self_loops.unsqueeze(-1)
        
        # 4. Áp dụng softmax và dropout
        # sparsemax = Sparsemax(dim=1)
        # attention = sparsemax(attention_logits)
        attention = F.softmax(attention_logits, dim=1)
        if torch.isnan(attention).any(): 
          print("NaN in attention! Logits min/max: ", attention.min(), attention.max())

        attention = self.dropout(attention) # (N, N, NH)

        # 5. Tổng hợp đặc trưng từ hàng xóm
        attention_permuted = attention.permute(2, 0, 1)
        nodes_features_proj_permuted = nodes_features_proj.permute(1, 0, 2)
        out_nodes_features_permuted = torch.bmm(attention_permuted, nodes_features_proj_permuted)
        out_nodes_features = out_nodes_features_permuted.permute(1, 0, 2)
        
        # 6. Nối hoặc lấy trung bình các head
        if self.concat:
            out_nodes_features = out_nodes_features.reshape(num_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        # if self.add_skip_connection:
        #     if out_nodes_features.shape[-1] == x.shape[-1]:
        #         out_nodes_features += x.unsqueeze(1) # Bỏ .view() vì x đã đúng shape
        #     else:
        #         out_nodes_features += self.skip_proj(x).view(-1, self.num_of_heads, self.num_out_features).mean(dim=1) if not self.concat else self.skip_proj(x)

        if self.bias is not None:
            out_nodes_features += self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)
        # print("out_nodes_features", out_nodes_features)
        return out_nodes_features

class GATEncoderDense(nn.Module):
    """
    Bộ mã hóa GAT hai lớp sử dụng GATConvDense.
    """
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()

        # self.norm_layer1 = nn.LayerNorm(in_features)

        self.gat_layer_1 = GATLayerDense(
            num_in_features=in_features,
            num_out_features=hidden_features,
            num_of_heads=num_heads,
            concat=True,
            activation=nn.ELU(),
            dropout_prob=dropout_prob
        )

        # self.activation1 = nn.ReLU()

        # self.norm_layer2 = nn.LayerNorm(hidden_features * num_heads)

        self.gat_layer_2 = GATLayerDense(
            num_in_features=hidden_features * num_heads,
            num_out_features=out_features,
            num_of_heads=1, # Thường lớp cuối cùng có 1 head hoặc avg
            concat=False,
            activation=None,
            dropout_prob=dropout_prob
        )

        # if in_features != hidden_features * num_heads:
        #   self.skip1 = nn.Linear(in_features, hidden_features * num_heads)
        # else:
        #   self.skip1 = nn.Identity()

        # if hidden_features * num_heads != out_features:
        #   self.skip2 = nn.Linear(hidden_features * num_heads, out_features)
        # else:
        #   self.skip2 = nn.Identity()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, adj):
      # PreNorm Architecture
      # Norm -> GAT -> Dropout -> Residual -> Activation -> Norm -> GAT -> Residual 
      # x_skip1 = self.skip1(x)

      # h = self.norm_layer1(x)
      # h = self.gat_layer_1(h, adj)
      # h = self.dropout(h)
      # x = x_skip1 + h 

      # x = self.activation1(x)

      # x_skip2 = self.skip2(x)
      # h = self.norm_layer2(x)
      # h = self.gat_layer_2(h, adj)
      # x = x_skip2 + h   
             
      # return x

      # A simpler architecture
      h = self.gat_layer_1(x, adj)
      h = self.dropout(h)
      h = self.gat_layer_2(h, adj)

      return h 

class NewGATLayerDense(nn.Module):
    """
    Phiên bản của lớp GAT có thể hoạt động với ma trận kề dày (dense).
    Điều này cho phép gradient chảy ngược qua ma trận kề,
    cần thiết cho các cuộc tấn công đối nghịch vào cấu trúc đồ thị.
    Phiên bản này hiện thực một cơ chế chú ý mới, dựa theo công thức tính hệ số chú ý từ paper scGAC
    """
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, bias=True):
        super().__init__()

        self.num_out_features = num_out_features
        self.num_of_heads = num_of_heads
        self.concat = concat

        # Ma trận trọng số W
        # self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # Vector chú ý a
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_ReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # import pdb; pdb.set_trace() 
        # x shape: (N, num_in_features), adj shape: (N, N)
        num_nodes = x.shape[0]

        if adj.is_sparse:
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj

        # --- CHÈN CODE KIỂM TRA VÀO ĐÂY ---
        # try:
        #     # 1. Tính tổng của mỗi hàng (dim=1)
        #     row_sums = dense_adj.sum(dim=1)
            
        #     # 2. Kiểm tra xem có hàng nào có tổng <= 0 không
        #     # (PGD có thể tạo ra số âm, nên ta kiểm tra <= 0 cho chắc)
        #     has_problematic_rows = torch.any(row_sums <= 0)
            
        #     if has_problematic_rows:
        #         problematic_indices = torch.where(row_sums <= 0)[0]
        #         print("="*50)
        #         print(f"!!! CẢNH BÁO TỪ GATConvDense !!!")
        #         print(f"Phát hiện {problematic_indices.numel()} hàng (node) có tổng <= 0 trong 'dense_adj'.")
        #         print(f"Đây là nguyên nhân GÂY RA NaN (log(0) hoặc 0/0).")
        #         print(f"Chỉ số của các hàng có vấn đề: {problematic_indices}")
        #         print("="*50)
        #         # Dừng chương trình để debug
        #         import pdb; pdb.set_trace() 
                
        # except Exception as e:
        #     print(f"Lỗi khi đang kiểm tra 'dense_adj': {e}")
        # --- KẾT THÚC CODE KIỂM TRA ---

        # Clamp value to 0-1
        adj_with_self_loops = dense_adj 
        # 1. Biến đổi đặc trưng tuyến tính
        nodes_features_proj = self.linear_proj(x).view(num_nodes, self.num_of_heads, self.num_out_features)
        if torch.isnan(nodes_features_proj).any(): 
          print("NaN in nodes_features_proj!")
        
        # 2. Tính toán điểm chú ý cho tất cả các cặp node
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        if torch.isnan(scores_source).any(): 
          print("NaN in scores_source!")
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        if torch.isnan(scores_target).any(): 
          print("NaN in scores_target!")
        # e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
        e = torch.exp(-torch.pow(scores_target.unsqueeze(1) - scores_source.unsqueeze(0), 2))
        if torch.isnan(e).any(): 
          print("NaN in e!")

        # zero_vec = -1e9 * torch.ones_like(e)

        additive_mask = (1.0 - adj_with_self_loops) * -1e9 
        attention_logits = e + additive_mask.unsqueeze(-1)

        
        # 4. Áp dụng softmax và dropout
        attention = F.softmax(attention_logits, dim=1)
        # attention = F.softmax(e, dim=1)
        if torch.isnan(attention).any(): 
          print("NaN in attention! Logits min/max: ", attention.min(), attention.max())

        attention = self.dropout(attention) # (N, N, NH)

        # 5. Tổng hợp đặc trưng từ hàng xóm
        attention_permuted = attention.permute(2, 0, 1)
        nodes_features_proj_permuted = nodes_features_proj.permute(1, 0, 2)
        out_nodes_features_permuted = torch.bmm(attention_permuted, nodes_features_proj_permuted)
        out_nodes_features = out_nodes_features_permuted.permute(1, 0, 2)

        # 6. Nối hoặc lấy trung bình các head
        if self.concat:
            out_nodes_features = out_nodes_features.reshape(num_nodes, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)
        # print("out_nodes_features", out_nodes_features)
        return out_nodes_features

class NewGATEncoderDense(nn.Module):
    """
    Bộ mã hóa GAT hai lớp sử dụng NewGATConvDense.
    """
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()

        # self.norm_layer1 = nn.LayerNorm(in_features)

        self.gat_layer_1 = NewGATLayerDense(
            num_in_features=in_features,
            num_out_features=hidden_features,
            num_of_heads=num_heads,
            concat=True,
            activation=nn.ReLU(),
            dropout_prob=dropout_prob
        )

        # self.activation1 = nn.ELU()

        # self.norm_layer2 = nn.LayerNorm(hidden_features * num_heads)

        self.gat_layer_2 = NewGATLayerDense(
            num_in_features=hidden_features * num_heads,
            num_out_features=out_features,
            num_of_heads=1, # Thường lớp cuối cùng có 1 head hoặc avg
            concat=False,
            activation=None,
            dropout_prob=dropout_prob
        )

        if in_features != hidden_features * num_heads:
          self.skip1 = nn.Linear(in_features, hidden_features * num_heads)
        else:
          self.skip1 = nn.Identity()

        if hidden_features * num_heads != out_features:
          self.skip2 = nn.Linear(hidden_features * num_heads, out_features)
        else:
          self.skip2 = nn.Identity()

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, adj):
      # PreNorm Architecture
      # Norm -> GAT -> Dropout -> Residual -> Activation -> Norm -> GAT -> Residual 
        # x_skip1 = self.skip1(x)

        # h = self.norm_layer1(x)
        # h = self.gat_layer_1(h, adj)
        # h = self.dropout(h)
        # x = x_skip1 + h 

        # x = self.activation1(x)

        # x_skip2 = self.skip2(x)
        # h = self.norm_layer2(x)
        # h = self.gat_layer_2(h, adj)
        # x = x_skip2 + h   
             
        # return x

      # A simpler architecture
      h = self.gat_layer_1(x, adj)
      h = self.dropout(h)
      h = self.gat_layer_2(h, adj)

      return h 

# from torch_geometric.nn import GATv2Conv

# class GATv2Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_features, num_heads=4, dropout_prob=0.6):
#         super(GATv2Encoder, self).__init__()
#         self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout_prob)
#         # Second GATv2Conv layer with concat=False to get desired output dimensions
#         self.conv2 = GATv2Conv(hidden_channels * num_heads, out_features, heads=1, concat=False, dropout=dropout_prob)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, x, adj):
#         if adj.is_sparse:
#             edge_index = adj._indices()

#         else:
#             edge_index = adj.nonzero().t().contiguous()

#         # First GAT layer with ReLU and dropout
#         x = F.elu(self.conv1(x, edge_index))
#         x = self.dropout(x)
#         # Second GAT layer
#         x = self.conv2(x, edge_index)
#         return x


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

        # --- ĐỊNH NGHĨA THAM SỐ MỘT LẦN DUY NHẤT ---
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
        # --- HẾT PHẦN ĐỊNH NGHĨA ---

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

    def sum_edge_scores_neighborhood(self, exp_scores_per_edge, trg_nodes_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_nodes_index_expanded = self.check_shape(trg_nodes_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sum = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sum.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sum.index_select(self.nodes_dim, trg_nodes_index)
    

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_nodes_index_expanded = self.check_shape(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)

        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_nodes_index_expanded, nodes_features_proj_lifted_weighted)

        return out_nodes_features


    def check_shape(self, this, other):
        # Append singleton dimension until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def forward(self, x, graph, edge_weight=None):
        # x shape = (N, Fin), graph có thể là (N, N) hoặc (2, E)
        num_nodes = x.shape[0]

        # x = self.dropout(x)

        # (N, Fin) -> (N, NH, Fout)
        nodes_features_proj = self.linear_proj(x).view(num_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        if graph.is_sparse or (graph.dim() == 2 and graph.shape[0] == 2):
            edge_index = graph
            if edge_index.is_sparse:
                edge_index = edge_index._indices()

            # if edge_weight is None:
            #     edge_weight = torch.ones(edge_index.size(1), device=x.device)
            
            # print("Edge weight", edge_weight)

            # edge_index, _ = remove_self_loops(edge_index)
            # edge_index, edge_weight = remove_self_loops(edge_index, edge_attr=edge_weight)
            
                # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            # if not contains_self_loops(edge_index):
            #     edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=num_nodes)


            #
            # Step 2: Edge attention calculation
            #

            # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
            # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
            # scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            # scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

            # # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
            # # the possible combinations of scores we just prepare those that will actually be used and those are defined
            # # by the edge index.
            # # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
            # src_nodes_index = edge_index[self.src_nodes_dim]
            # trg_nodes_index = edge_index[self.trg_nodes_dim]

            # # Note: Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
            # scores_source_lifted = scores_source.index_select(self.nodes_dim, src_nodes_index)
            # scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_nodes_index)
            # nodes_features_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)

            # # Compute the attention coefficients for each edge
            
            # # As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
            # # Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
            # # into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
            # # in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
            # # (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
            # #  i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

            # # Note:
            # # Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
            # # and it's a fairly common "trick" used in pretty much every deep learning framework.
            # # Check out this link for more details:
            # # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

            # scores_per_edge = self.leaky_ReLU(scores_source_lifted + scores_target_lifted)
            # # print("scores_per_edge before masking:", scores_per_edge)

            # # m = torch.rand_like(edge_weight)
            # # edge_weight_binary = (edge_weight > 0).float()
            # # edge_weight_differentiable = edge_weight_binary - edge_weight.detach() + edge_weight
            # # mask_value = (1.0 - edge_weight_differentiable.view(-1, 1)) * -1e9

            # # mask_value = (1.0 - edge_weight.view(-1, 1)) * -100

            # # mask_value = torch.log(edge_weight.view(-1, 1) + 1e-9)

            # # scores_per_edge = scores_per_edge + mask_value

            # # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
            # # scores_per_edge = scores_per_edge - scores_per_edge.max()
            # # exp_scores_per_edge = scores_per_edge.exp()

            # # Calculate the denominator. Shape = (E, NH)
            # # neighborhood_sum_denominator = self.sum_edge_scores_neighborhood(exp_scores_per_edge, edge_index[self.trg_nodes_dim], num_nodes)

            # # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
            # # possibility of the computer rounding a very small number all the way to 0.
            # # attention_per_edge = exp_scores_per_edge / (neighborhood_sum_denominator + 1e-16)
            # attention_per_edge = softmax(scores_per_edge, index=trg_nodes_index, num_nodes=num_nodes)
            # attention_per_edge = attention_per_edge.unsqueeze(-1)
            # attention_per_edge = self.dropout(attention_per_edge)


            # #
            # # Step 3: Neighborhood aggregation
            # #

            # # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
            # # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
            # nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attention_per_edge

            # # This part sums up weighted and projected neighborhood feature vectors for every target node
            # # shape = (N, NH, FOUT)
            # out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, x, num_nodes)

            # --------------------------------------------------------------------------

            scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

            # Lấy features của các cặp cạnh (E, NH)
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
                # edge_index, 
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
                # Lấy alpha của head hiện tại: (E,)
                alpha_h = attention_per_edge[:, head]
                
                # Tạo Sparse Matrix (Adj có trọng số) cho head này
                # Shape (N, N)
                adj_sparse_h = torch.sparse_coo_tensor(
                    attention_indices, 
                    alpha_h, 
                    (num_nodes, num_nodes)
                )
                adj_sparse_h = adj_sparse_h.coalesce()
                # Lấy features của head hiện tại: (N, Fout)
                h_prime = nodes_features_proj[:, head, :]
                
                # Thực hiện phép nhân ma trận Sparse x Dense
                # (N, N) x (N, Fout) -> (N, Fout)
                # Đây chính là operation tương đương nhất với bmm
                out_h = torch.sparse.mm(adj_sparse_h, h_prime)
                outputs_per_head.append(out_h)
            
            # Gom lại: (N, NH, Fout)
            out_nodes_features = torch.stack(outputs_per_head, dim=1)

            # ----------------------------------------------------------------

            # adj = to_dense_adj(graph, max_num_nodes= x.shape[0])[0] # Đây là ma trận kề (N, N)
            # adj = torch.clamp(adj, 0, 1) 
            # # adj = adj.t()
            
            # # (N, NH)
            # # scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            # # scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

            # scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
            # scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

            # scores_target = scores_target.transpose(0, 1)
            # scores_source = scores_source.permute(1, 2, 0)

            # # # scores_source = scores_source.transpose(0, 1)
            # # # scores_target = scores_target.permute(1, 2, 0)
            
            # # # (N, N, NH) - Tính e_ij cho TẤT CẢ các cặp
            # # # e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
            # # # e = self.leaky_ReLU(scores_source.unsqueeze(1) + scores_target.unsqueeze(0))
            # all_scores = self.leaky_ReLU(scores_target + scores_source)
            # # # all_scores = self.leaky_ReLU(scores_source + scores_target)

            # # # Masking
            # additive_mask = (1.0 - adj) * -1e9
            # attention_logits = all_scores + additive_mask.unsqueeze(0)

            # # # all_attention_coefficients = self.softmax(attention_logits)
            
            # # # Softmax
            # all_attention_coefficients = F.softmax(attention_logits, dim=-1) # (N, N, NH)

            # all_attention_coefficients = self.dropout(all_attention_coefficients)

            # # # Aggregate
            # # # (NH, N, N) x (NH, N, Fout) -> (NH, N, Fout)
            # # # out_nodes_features = torch.bmm(all_attention_coefficients.permute(2, 0, 1), nodes_features_proj.permute(1, 0, 2))
            # # # out_nodes_features = out_nodes_features.permute(1, 0, 2) # (N, NH, Fout)
            # out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
            # out_nodes_features = out_nodes_features.permute(1, 0, 2)

            # scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            # scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
            
            # # (N, N, NH) - Tính e_ij cho TẤT CẢ các cặp
            # e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
            # # e = self.leaky_ReLU(scores_source.unsqueeze(1) + scores_target.unsqueeze(0))

            # # Masking
            # additive_mask = (1.0 - adj) * -1e9
            # attention_logits = e + additive_mask.unsqueeze(-1)
            
            # # Softmax
            # attention = F.softmax(attention_logits, dim=1) # (N, N, NH)
            # # attention = self.softmax(attention_logits) # (N, N, NH)
            # attention = self.dropout(attention)

            # # Aggregate
            # # (NH, N, N) x (NH, N, Fout) -> (NH, N, Fout)
            # out_nodes_features_permuted = torch.bmm(attention.permute(2, 0, 1), nodes_features_proj.permute(1, 0, 2))
            # out_nodes_features = out_nodes_features_permuted.permute(1, 0, 2) # (N, NH, Fout)
        elif graph.dim() == 2 and graph.shape[0] == graph.shape[1]:
            adj = graph # Đây là ma trận kề (N, N)
            # adj = adj.t()
            
            # (N, NH)
            # scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            # scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

            scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
            scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)

            scores_target = scores_target.transpose(0, 1)
            scores_source = scores_source.permute(1, 2, 0)

            # scores_source = scores_source.transpose(0, 1)
            # scores_target = scores_target.permute(1, 2, 0)
            
            # (N, N, NH) - Tính e_ij cho TẤT CẢ các cặp
            # e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
            # e = self.leaky_ReLU(scores_source.unsqueeze(1) + scores_target.unsqueeze(0))
            all_scores = self.leaky_ReLU(scores_target + scores_source)
            # all_scores = self.leaky_ReLU(scores_source + scores_target)

            # Masking
            additive_mask = (1.0 - adj) * -1e9
            attention_logits = all_scores + additive_mask.unsqueeze(0)

            # all_attention_coefficients = self.softmax(attention_logits)
            
            # Softmax
            all_attention_coefficients = F.softmax(attention_logits, dim=-1) # (N, N, NH)

            all_attention_coefficients = self.dropout(all_attention_coefficients)

            # Aggregate
            # (NH, N, N) x (NH, N, Fout) -> (NH, N, Fout)
            # out_nodes_features = torch.bmm(all_attention_coefficients.permute(2, 0, 1), nodes_features_proj.permute(1, 0, 2))
            # out_nodes_features = out_nodes_features.permute(1, 0, 2) # (N, NH, Fout)
            out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
            out_nodes_features = out_nodes_features.permute(1, 0, 2)

            # adj = graph # Đây là ma trận kề (N, N)
            # # adj = adj.t()
            
            # # (N, NH)
            # scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
            # scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
            
            # # (N, N, NH) - Tính e_ij cho TẤT CẢ các cặp
            # e = self.leaky_ReLU(scores_target.unsqueeze(1) + scores_source.unsqueeze(0))
            # # e = self.leaky_ReLU(scores_source.unsqueeze(1) + scores_target.unsqueeze(0))

            # # Masking
            # additive_mask = (1.0 - adj) * -1e9
            # attention_logits = e + additive_mask.unsqueeze(-1)
            
            # # Softmax
            # attention = F.softmax(attention_logits, dim=1) # (N, N, NH)
            # # attention = self.softmax(attention_logits) # (N, N, NH)
            # attention = self.dropout(attention)

            # # Aggregate
            # # (NH, N, N) x (NH, N, Fout) -> (NH, N, Fout)
            # out_nodes_features_permuted = torch.bmm(attention.permute(2, 0, 1), nodes_features_proj.permute(1, 0, 2))
            # out_nodes_features = out_nodes_features_permuted.permute(1, 0, 2) # (N, NH, Fout)
            
        else:
            raise ValueError("Đầu vào không xác định: không phải Adj (N,N) hay edge_index (2,E).")

        # --- BƯỚC 3: KẾT HỢP (Chung cho cả hai) ---
        # Skip connection 
        # if self.add_skip_connection:  # add skip or residual connection
        #     if out_nodes_features.shape[-1] == x.shape[-1]:  # if FIN == FOUT
        #         # print("FIN == FOUT, using simple skip connection")
        #         # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
        #         # thus we're basically copying input vectors NH times and adding to processed vectors
        #         out_nodes_features += x.unsqueeze(1)
        #     else:
        #         # print("FIN != FOUT, using skip projection")
        #         # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
        #         # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
        #         out_nodes_features += self.skip_proj(x).view(-1, self.num_of_heads, self.num_out_features)

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
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob_1, dropout_prob_2):
        super().__init__()

        print("Infeatures_dim: ", in_features)
        print("Hidden_dim: ", hidden_features)
        print("Outfeatures_dim: ", out_features)

        self.gat_layer_1 = GATLayerHybrid(
            num_in_features= in_features,
            num_out_features= hidden_features,
            num_of_heads= num_heads,
            concat= True,
            activation= nn.ReLU(),
            # activation= nn.ELU(),
            dropout_prob= dropout_prob_1,
            add_skip_connection=False #
        )

        self.dropout = nn.Dropout(dropout_prob_1)

        self.gat_layer_2 = GATLayerHybrid(
            num_in_features= hidden_features * num_heads,
            num_out_features= out_features,
            num_of_heads= 1,
            concat= False,
            activation= None,
            dropout_prob= dropout_prob_1,
            add_skip_connection=False
        )

    def forward(self, x, graph, edge_weight=None):
        # 'graph' có thể là adj (N, N) hoặc edge_index (2, E)
        x = self.gat_layer_1(x, graph, edge_weight)
        x = self.dropout(x)
        x = self.gat_layer_2(x, graph, edge_weight)
        return x


class UnifiedGATLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True):
        super(UnifiedGATLayer, self).__init__()
        self.in_features = num_in_features
        self.out_features = num_out_features
        self.heads = num_of_heads
        self.concat = concat
        self.dropout = dropout_prob
        self.add_skip_connection = add_skip_connection
        self.bias = bias
        self.activation = activation

        # Linear Transformation
        self.lin = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # Attention Mechanisms
        self.att_src = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [N, in_features]
        edge_index: [2, E]
        edge_weight: [E] - Đây là chìa khóa để tấn công! (Optional)
        """
        N = x.size(0)
        
        # 1. Linear Projection
        x_proj = self.lin(x).view(-1, self.heads, self.out_features)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        # 2. Add self-loops (Bắt buộc cho GAT để giữ thông tin node hiện tại)
        if edge_weight is None:
            # Trọng số mặc định là 1 cho tất cả các cạnh
            edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        else:
            # Nếu có edge_weight (từ tấn công), ta giả định edge_index đã bao gồm self-loops
            # hoặc người gọi phải tự xử lý việc add self-loop có trọng số.
            pass

        # 3. Compute Attention Scores
        # Lấy feature của node nguồn và đích
        # x_src: [E, heads, out_features]
        x_src = x_proj[edge_index[0]]
        x_dst = x_proj[edge_index[1]]

        # Tính Score: (x_src * att_src).sum + (x_dst * att_dst).sum
        # alpha: [E, heads]
        alpha = (x_src * self.att_src).sum(dim=-1) + (x_dst * self.att_dst).sum(dim=-1)
        alpha = self.leaky_relu(alpha)

        # 4. Normalize Attention (Softmax)
        # alpha sẽ được normalize theo từng node đích (target node)
        alpha = softmax(alpha, edge_index[1], num_nodes=N)
        
        # 5. INJECTION: Đưa trọng số tấn công (Adversarial Weights) vào
        # Thay vì nhân ma trận Dense, ta nhân trực tiếp vào attention score
        # A_ij * alpha_ij
        # edge_weight shape [E] -> [E, 1] để broadcase với heads
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        # Dropout lên attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 6. Aggregate Features
        # out: [N, heads, out_features]
        # Weighted sum: out_i = sum(alpha_ij * x_j)
        out_nodes_features = torch.zeros(N, self.heads, self.out_features, device=x.device)
        # scatter_add_ requires `out` and `src` to handle dimensions correctly
        # Ta dùng loop hoặc function hỗ trợ message passing. 
        # Để đơn giản và nhanh, ta dùng cách thủ công tính weighted value rồi scatter:
        
        weighted_nodes = x_src * alpha.unsqueeze(-1) # [E, heads, out]
        
        # Dùng index_add_ hoặc scatter_add
        # Target index là edge_index[1] (dòng chảy từ 0 -> 1)
        # Tuy nhiên quy ước GAT thường là j -> i (source -> target). Ở đây 0->src, 1->dst.
        # PyG scatter convention: dim size check
        index = edge_index[1].unsqueeze(-1).unsqueeze(-1).expand_as(weighted_nodes)
        out_nodes_features.scatter_add_(0, index, weighted_nodes)

        # 7. Concatenate or Mean
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.heads * self.out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features = out_nodes_features + self.bias

        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)

        return out_nodes_features

class UnifiedGATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_prob):
        super().__init__()
        self.layer1 = UnifiedGATLayer(
            num_in_features= in_features,
            num_out_features= hidden_features,
            num_of_heads= num_heads,
            concat= True,
            activation= nn.ReLU(),
            dropout_prob= dropout_prob,
            add_skip_connection=False
        )
        
        self.layer2 = UnifiedGATLayer(
            num_in_features= hidden_features * num_heads,
            num_out_features= out_features,
            num_of_heads= 1,
            concat= False,
            activation= None,
            dropout_prob= dropout_prob,
            add_skip_connection=False
        )
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index, edge_weight=None):
        # Layer 1
        x = self.layer1(x, edge_index, edge_weight)
        # x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x, edge_index, edge_weight) 
        return x