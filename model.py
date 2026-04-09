import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import  add_self_loops, dense_to_sparse, degree, remove_self_loops
# from utils import GCNConv

# class Encoder(torch.nn.Module):
#     def __init__(self, dim_in: int, dim_out: int, num_layers: int = 2):
#         super(Encoder, self).__init__()
#         self.base_model = GCNConv
#         self.activation = F.relu
#         assert num_layers >= 2
#         self.num_layers= num_layers
#         self.conv = [GCNConv(dim_in, 2 * dim_out)]
#         for _ in range(1, num_layers-1):
#             self.conv.append(GCNConv(2 * dim_out, 2 * dim_out))
#         self.conv.append(GCNConv(2 * dim_out, dim_out))
#         self.conv = nn.ModuleList(self.conv)


#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         for i in range(self.num_layers):
#             x = self.activation(self.conv[i](x, edge_index))
#         return x
    
  
    
# class AGCLModel(torch.nn.Module):
#     def __init__(self, encoder: Encoder, n_hidden: int, n_proj_hidden: int,
#                  tau: float = 0.5):
#         super(AGCLModel, self).__init__()
#         self.encoder: Encoder = encoder
#         self.tau: float = tau

#         self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
#         self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)
      
        
#     def forward(self, x: torch.Tensor,
#                 Adj: torch.Tensor) -> torch.Tensor:
       
#         return self.encoder(x, Adj)

#     def projection(self, z: torch.Tensor) -> torch.Tensor:
#         z = F.elu(self.fc_layer1(z))
#         return self.fc_layer2(z)

#     #Codes are modified from https://github.com/Shengyu-Feng/ARIEL
#     def sim(self, z1: torch.Tensor, z2: torch.Tensor):
#         z1 = F.normalize(z1)
#         z2 = F.normalize(z2)
#         return torch.mm(z1, z2.t())

#     def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
#         f = lambda x: torch.exp(x / self.tau)
#         refl_sim = f(self.sim(z1, z1))
#         between_sim = f(self.sim(z1, z2))

#         return -torch.log(
#             between_sim.diag()
#             / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

#     def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
#                           batch_size: int):
#         # Space complexity: O(BN) (semi_loss: O(N^2))
#         device = z1.device
#         num_nodes = z1.size(0)
#         num_batches = (num_nodes - 1) // batch_size + 1
#         f = lambda x: torch.exp(x / self.tau)
#         indices = torch.arange(0, num_nodes).to(device)
#         losses = []

#         for i in range(num_batches):
#             mask = indices[i * batch_size:(i + 1) * batch_size]
#             refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
#             between_sim = f(self.sim(z1[mask], z2))  # [B, N]

#             losses.append(-torch.log(
#                 between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
#                 / (refl_sim.sum(1) + between_sim.sum(1)
#                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

#         return torch.cat(losses)

#     def loss(self, z1: torch.Tensor, z2: torch.Tensor,
#              mean: bool = True, batch_size: int = 0):
#         h1 = self.projection(z1)
#         h2 = self.projection(z2)
        
            
#         if batch_size == 0:
#             l1 = self.semi_loss(h1, h2)
#             l2 = self.semi_loss(h2, h1)
#         else:
#             l1 = self.batched_semi_loss(h1, h2, batch_size)
#             l2 = self.batched_semi_loss(h2, h1, batch_size)

#         ret = (l1 + l2) * 0.5
#         ret = ret.mean() 

#         return ret

from GATEncoder import GATEncoderDense, NewGATEncoderDense, GATEncoder, GATEncoderHybrid, UnifiedGATEncoder
class NewModel(torch.nn.Module):
    def __init__(self, encoder: GATEncoder, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(NewModel, self).__init__()
        self.encoder: GATEncoder = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        # self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        # self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

    # def fuse(self, z: torch.Tensor):
    #     h = F.elu(self.fuse_fc1(z))
    #     return self.fuse_fc2(h)  # back to n_hidden

    # def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:

    #     return self.encoder(x, Adj)

    def forward(self, x: torch.Tensor, edge_ind: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        return self.encoder(x, edge_ind, edge_weight)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2
    

class NewModel2(torch.nn.Module):
    def __init__(self, encoder: GATEncoderDense, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(NewModel2, self).__init__()
        self.encoder: GATEncoderDense = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

    def fuse(self, z: torch.Tensor):
        h = F.elu(self.fuse_fc1(z))
        return self.fuse_fc2(h)  # back to n_hidden

    def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:

        return self.encoder(x, Adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2

class HybridGATModel(torch.nn.Module):
    def __init__(self, encoder: GATEncoderHybrid, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(HybridGATModel, self).__init__()
        self.encoder: GATEncoderHybrid = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        # self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        # self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

        # self.bn = nn.BatchNorm1d(n_hidden)

    # def fuse(self, z: torch.Tensor):
    #     h = F.elu(self.fuse_fc1(z))
    #     return self.fuse_fc2(h)  # back to n_hidden

    # def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:
    #     return self.encoder(x, Adj)
    #     # return self.bn(self.encoder(x, Adj))

    def forward(self, x: torch.Tensor, edge_ind: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        return self.encoder(x, edge_ind, edge_weight)
        # return self.bn(self.encoder(x, Adj))

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def sim(self, z1: torch.Tensor, z2: torch.Tensor):
    #     # Tính khoảng cách Euclidean bình phương giữa mọi cặp (Pairwise Distance)
    #     # z1: [N, D], z2: [N, D] -> Output: [N, N]
    #     # Công thức: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        
    #     z1_sq = torch.sum(z1**2, dim=1, keepdim=True)
    #     z2_sq = torch.sum(z2**2, dim=1, keepdim=True)
        
    #     # 2. Tính tích vô hướng (Dot Product)
    #     # prod: [N, N]
    #     prod = torch.mm(z1, z2.t())
        
    #     # 3. Áp dụng hằng đẳng thức: a^2 + b^2 - 2ab
    #     # Broadcasting sẽ tự lo phần kích thước: [N, 1] + [1, N] - [N, N] -> [N, N]
    #     dist_sq = z1_sq + z2_sq.t() - 2 * prod
        
    #     # 4. Quan trọng: Kẹp giá trị (Clamp) để tránh sai số máy tính ra số âm (vd: -0.000001)
    #     dist_sq = torch.clamp(dist_sq, min=1e-6)

    #     d = z1.shape[1]

    #     scale_factor = d ** 0.5
        
    #     # Chuyển thành Similarity: Dùng dấu ÂM
    #     # Càng gần -> khoảng cách càng nhỏ -> sim càng lớn (gần 0)
    #     # Càng xa -> khoảng cách càng lớn -> sim càng nhỏ (âm vô cùng)
    #     return -dist_sq / scale_factor

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2


from GNNEncoder import SimpleGNNEncoder
class SimpleGNNModel(torch.nn.Module):
    def __init__(self, encoder: SimpleGNNEncoder, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(SimpleGNNModel, self).__init__()
        self.encoder: SimpleGNNEncoder = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        # self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        # self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

        # self.bn = nn.BatchNorm1d(n_hidden)

    # def fuse(self, z: torch.Tensor):
    #     h = F.elu(self.fuse_fc1(z))
    #     return self.fuse_fc2(h)  # back to n_hidden

    # def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:
    #     return self.encoder(x, Adj)
    #     # return self.bn(self.encoder(x, Adj))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)
        # return self.bn(self.encoder(x, Adj))

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def sim(self, z1: torch.Tensor, z2: torch.Tensor):
    #     # Tính khoảng cách Euclidean bình phương giữa mọi cặp (Pairwise Distance)
    #     # z1: [N, D], z2: [N, D] -> Output: [N, N]
    #     # Công thức: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        
    #     z1_sq = torch.sum(z1**2, dim=1, keepdim=True)
    #     z2_sq = torch.sum(z2**2, dim=1, keepdim=True)
        
    #     # 2. Tính tích vô hướng (Dot Product)
    #     # prod: [N, N]
    #     prod = torch.mm(z1, z2.t())
        
    #     # 3. Áp dụng hằng đẳng thức: a^2 + b^2 - 2ab
    #     # Broadcasting sẽ tự lo phần kích thước: [N, 1] + [1, N] - [N, N] -> [N, N]
    #     dist_sq = z1_sq + z2_sq.t() - 2 * prod
        
    #     # 4. Quan trọng: Kẹp giá trị (Clamp) để tránh sai số máy tính ra số âm (vd: -0.000001)
    #     dist_sq = torch.clamp(dist_sq, min=1e-6)

    #     d = z1.shape[1]

    #     scale_factor = d ** 0.5
        
    #     # Chuyển thành Similarity: Dùng dấu ÂM
    #     # Càng gần -> khoảng cách càng nhỏ -> sim càng lớn (gần 0)
    #     # Càng xa -> khoảng cách càng lớn -> sim càng nhỏ (âm vô cùng)
    #     return -dist_sq / scale_factor

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2


class Finetune_Model(NewModel2):
    def __init__(self, pretrained_model: NewModel2, num_hidden: int, num_proj_hidden: int, num_clusters: int,  alpha: float = 1.0,
                 tau: float = 0.4):
        super().__init__(encoder=pretrained_model.encoder, n_hidden=num_hidden, n_proj_hidden=num_proj_hidden, tau=tau)
    
        self.fc_layer1 = pretrained_model.fc_layer1 
        self.fc_layer2 = pretrained_model.fc_layer2
        self.fuse_fc1 = pretrained_model.fuse_fc1
        self.fuse_fc2 = pretrained_model.fuse_fc2

        self.alpha: float = alpha  # Parameter for the t-distribution
        self.num_clusters = num_clusters
        self.cluster_centers = torch.randn(num_clusters, num_hidden, requires_grad=False)
        # initial_centers = torch.randn(num_clusters, num_hidden) 
        # self.cluster_centers = torch.nn.Parameter(initial_centers)

    def calculate_q(self, Z: torch.Tensor):
      cluster_centers = self.cluster_centers
      cluster_centers = cluster_centers.to(Z.device)
      # print('Z normalize: ', F.normalize(Z, p = 2, dim = 1))
      # print('cluster_centers: ', F.normalize(cluster_centers, p = 2, dim = 1))
      dis = torch.cdist(Z, cluster_centers)
    #   dis = torch.cdist(F.normalize(Z, p = 2, dim = 1), F.normalize(cluster_centers, p = 2, dim = 1))
      # Soft assignments (q) using Student's t-distribution
      q = 1.0 / (1.0 + dis**2 / self.alpha)
      q = q ** ((self.alpha + 1.0) / 2.0)
      q = q / (q.sum(dim=1, keepdim=True))  # Normalize to probabilities

      return q 


    def calculate_p(self, q):
      f = torch.sum(q, dim=0)
      p = (q**2) / (f)
      p =  p / (p.sum(dim=1, keepdim=True))
      return p.detach()


    def clustering_loss_new(self, Z):
      q = self.calculate_q(Z)
      p = self.calculate_p(q)
      q_log = torch.log(q)
      kl = F.kl_div(q_log, p, reduction='batchmean')

      if kl < 0:  
        kl = torch.tensor(0.0, dtype=torch.float64, device=p.device)
      
      # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
      # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
      return kl

      # print(f'Value of q: {q}')
      # print(f'Value of p: {p}')

      # kl = torch.sum(p * (torch.log(p) - torch.log(q)))
      # kl = kl / p.size(0)
      # return F.kl_div(torch.log(q), p, reduction='batchmean')
      # return kl  

    def clustering_loss_new1(self, q, p):
      p_target = p.detach()

    #   confidence_threshold = 0.9

    #   max_prob, _ = torch.max(p_target, dim=1)
    #   confident_mask = max_prob > confidence_threshold
    #   q_confident = q[confident_mask]
    #   p_confident = p_target[confident_mask]
    #   if q_confident.shape[0] == 0:
    #     return torch.tensor(0.0, device=q.device)
      q_log = torch.log(q + 1e-9)
      kl = F.kl_div(q_log, p_target, reduction='batchmean')
    #   q_log_confident = torch.log(q_confident + 1e-9)
    #   kl = F.kl_div(q_log_confident, p_confident, reduction='batchmean')

      if kl < 0:  
        kl = torch.tensor(0.0, dtype=torch.float64, device=p.device)
      
      # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
      # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
      return kl

    def update_clusters_center(self, Z, q, num_cluster, device):
      # cluster_sizes = torch.sum(q, dim=0)
      # print('Cluster_sizes: ', cluster_sizes)
      Z = F.normalize(Z, p=2, dim=1)
      q_trans = q.t()
      clusters_center_updated = torch.matmul(q_trans, Z) / (torch.sum(q, dim=0).unsqueeze(1))
      # clusters_center_updated = torch.matmul(q, Z) / (torch.sum(q, dim=0).unsqueeze(1) + 1e-8)      
      self.cluster_centers = F.normalize(clusters_center_updated, p=2, dim=1)

# class HybridGATFinetune_Model(HybridGATModel):
#     def __init__(self, pretrained_model: HybridGATModel, num_hidden: int, num_proj_hidden: int, num_clusters: int,  alpha: float = 1.0,
#                  tau: float = 0.4, kappa: float = 20.0):
#         super().__init__(encoder=pretrained_model.encoder, n_hidden=num_hidden, n_proj_hidden=num_proj_hidden, tau=tau)
    
#         self.fc_layer1 = pretrained_model.fc_layer1 
#         self.fc_layer2 = pretrained_model.fc_layer2
#         self.fuse_fc1 = pretrained_model.fuse_fc1
#         self.fuse_fc2 = pretrained_model.fuse_fc2

#         self.alpha: float = alpha  # Parameter for the t-distribution
#         # self.kappa: float = kappa  # Parameter for von Mises-Fisher distribution
#         self.num_clusters = num_clusters
#         # self.cluster_centers = torch.randn(num_clusters, num_hidden, requires_grad=False)
#         self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, num_hidden))
#         torch.nn.init.xavier_normal_(self.cluster_centers.data)
#         # initial_centers = torch.randn(num_clusters, num_hidden) 
#         # self.cluster_centers = torch.nn.Parameter(initial_centers)

#         # self.cluster_layer = nn.Linear(num_hidden, num_clusters)
        
#         # # Khởi tạo trực giao (Orthogonal) giúp Spectral Clustering hội tụ nhanh hơn
#         # nn.init.orthogonal_(self.cluster_layer.weight)

#     def calculate_q(self, Z: torch.Tensor):
#       cluster_centers = self.cluster_centers
#       cluster_centers = cluster_centers.to(Z.device)
#       # print('Z normalize: ', F.normalize(Z, p = 2, dim = 1))
#     #   print('cluster_centers: ', F.normalize(cluster_centers, p = 2, dim = 1))
#       dis = torch.cdist(Z, cluster_centers)
#     #   dis = torch.cdist(F.normalize(Z, p = 2, dim = 1), F.normalize(cluster_centers, p = 2, dim = 1))
#       # Soft assignments (q) using Student's t-distribution
#       q = 1.0 / (1.0 + dis**2 / self.alpha)
#       q = q ** ((self.alpha + 1.0) / 2.0)
#       q = q / (q.sum(dim=1, keepdim=True))  # Normalize to probabilities

#       return q 

#     # def calculate_q(self, Z: torch.Tensor):
#     #     """
#     #     Tính soft assignment Q dựa trên vMF kernel (Cosine Similarity).
#     #     Công thức: q_ij = exp(kappa * cos(z_i, mu_j)) / sum_k(...)
#     #     """
#     #     # A. Normalize Z và Tâm cụm về mặt cầu đơn vị (Hypersphere)
#     #     # Đây là yêu cầu bắt buộc của vMF/Contrastive Learning [cite: 2650, 2738]
#     #     Z_norm = F.normalize(Z, p=2, dim=1)
        
#     #     cluster_centers = self.cluster_centers.to(Z.device)
#     #     centers_norm = F.normalize(cluster_centers, p=2, dim=1)

#     #     # B. Tính Cosine Similarity (Dot Product)
#     #     # (N, D) x (D, K) -> (N, K)
#     #     logits = torch.matmul(Z_norm, centers_norm.t())

#     #     # C. Scale bằng kappa (Concentration parameter)
#     #     # Kappa càng lớn -> Phân phối càng nhọn (cụm càng gọn)
#     #     logits = logits * self.kappa 

#     #     # D. Softmax để ra xác suất
#     #     q = F.softmax(logits, dim=1)

#     #     return q


#     def calculate_p(self, q):
#       f = torch.sum(q, dim=0)
#       p = (q**2) / (f)
#       p =  p / (p.sum(dim=1, keepdim=True))
#       return p.detach()


#     def clustering_loss_new(self, Z):
#       q = self.calculate_q(Z)
#       p = self.calculate_p(q)
#       q_log = torch.log(q)
#       kl = F.kl_div(q_log, p, reduction='batchmean')

#       if kl < 0:  
#         kl = torch.tensor(0.0, dtype=torch.float64, device=p.device)
      
#       # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
#       # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
#       return kl

#       # print(f'Value of q: {q}')
#       # print(f'Value of p: {p}')

#       # kl = torch.sum(p * (torch.log(p) - torch.log(q)))
#       # kl = kl / p.size(0)
#       # return F.kl_div(torch.log(q), p, reduction='batchmean')
#       # return kl  

#     def clustering_loss_new1(self, q, p):
#       p_target = p.detach()

#     #   confidence_threshold = 0.9

#     #   max_prob, _ = torch.max(p_target, dim=1)
#     #   confident_mask = max_prob > confidence_threshold
#     #   q_confident = q[confident_mask]
#     #   p_confident = p_target[confident_mask]
#     #   if q_confident.shape[0] == 0:
#     #     return torch.tensor(0.0, device=q.device)
#       q_log = torch.log(q + 1e-9)
#       kl = F.kl_div(q_log, p_target, reduction='batchmean')
#     #   q_log_confident = torch.log(q_confident + 1e-9)
#     #   kl = F.kl_div(q_log_confident, p_confident, reduction='batchmean')

#       kl = torch.nn.functional.relu(kl)
      
#       # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
#       # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
#       return kl

#     def update_clusters_center(self, Z, q, num_cluster, device):
#       # cluster_sizes = torch.sum(q, dim=0)
#       # print('Cluster_sizes: ', cluster_sizes)
#     #   Z = F.normalize(Z, p=2, dim=1)
#       q_trans = q.t()
#       clusters_center_updated = torch.matmul(q_trans, Z) / (torch.sum(q, dim=0).unsqueeze(1))
#       # clusters_center_updated = torch.matmul(q, Z) / (torch.sum(q, dim=0).unsqueeze(1) + 1e-8)      
#     #   self.cluster_centers = F.normalize(clusters_center_updated, p=2, dim=1)
#       self.cluster_centers = clusters_center_updated

#     def forward_cluster(self, x, adj):
#         """
#         Trả về Ma trận gán nhãn mềm Y (N x K)
#         """
#         z = self.forward(x, adj) # Lấy Z từ Encoder + BatchNorm
        
#         # Qua lớp linear -> Softmax để ra xác suất
#         # (N, Hidden) -> (N, K)
#         y_logits = self.cluster_layer(z)
#         y_soft = F.softmax(y_logits, dim=1)
#         return y_soft, z

#     def spectral_loss_sparse(self, Y, edge_index):
#         """
#         Tính Deep Spectral Clustering Loss (MinCut) trực tiếp trên Edge Index (Sparse).
#         Tiết kiệm bộ nhớ, chạy được cho dataset lớn.
        
#         Y: [N, K] - Soft Assignment Matrix
#         edge_index: [2, E] - Danh sách cạnh (nên là full graph symmetric)
#         """
#         num_nodes = Y.shape[0]
#         device = Y.device

#         # 1. Thêm Self-loops (Quan trọng để giữ tính ổn định cho Laplacian)
#         edge_index, _ = remove_self_loops(edge_index)
        
#         edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
#         # 2. Tính toán Ma trận kề chuẩn hóa (Normalized Adjacency)
#         # Công thức: A_norm = D^-1/2 * A * D^-1/2
#         row, col = edge_index_sl
        
#         # Tính bậc (Degree) của mỗi node
#         deg = degree(row, num_nodes, dtype=Y.dtype)
        
#         # Tính D^-1/2
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # Xử lý chia cho 0
        
#         # Tính trọng số cho từng cạnh trong edge_index
#         # edge_weight_norm = 1 / sqrt(deg_i * deg_j)
#         edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
#         # 3. Tạo Ma trận Thưa (Sparse Tensor) biểu diễn A_norm
#         adj_sparse = torch.sparse_coo_tensor(
#             edge_index_sl, 
#             edge_weight, 
#             (num_nodes, num_nodes)
#         ).to(device)
        
#         # 4. Tính MinCut Loss = -Trace(Y.T * A_norm * Y) / N
        
#         # Bước 4a: Nhân Sparse x Dense: (N, N) x (N, K) -> (N, K)
#         # PyTorch hỗ trợ tốt phép nhân này (spmm)
#         A_Y = torch.sparse.mm(adj_sparse, Y)
        
#         # Bước 4b: Nhân Dense x Dense: (K, N) x (N, K) -> (K, K)
#         Y_t_A_Y = torch.matmul(Y.t(), A_Y)
        
#         # Bước 4c: Tính Trace (Tổng đường chéo)
#         # Chia cho num_nodes để loss không quá lớn với data lớn
#         mincut_loss = -torch.trace(Y_t_A_Y) / num_nodes
        
#         # 5. Orthogonality Constraint (Tránh suy biến - Collapsing)
#         # Yêu cầu: (Y.T * Y) tiến tới I (Ma trận đơn vị)
#         # Điều này ép các cụm phải có kích thước xấp xỉ nhau và không chồng lấn.
        
#         # Normalize Y theo cột để ổn định
#         Y_norm = F.normalize(Y, p=2, dim=0) 
#         I = torch.eye(self.num_clusters, device=device)
        
#         ortho_loss = torch.norm(torch.matmul(Y_norm.t(), Y_norm) - I)
        
#         # Tổng hợp Loss (Lambda thường là 1.0 hoặc 2.0)
#         return mincut_loss + 2.0 * ortho_loss

class HybridGATFinetune_Model(HybridGATModel):
    def __init__(self, pretrained_model: HybridGATModel, num_hidden: int, num_proj_hidden: int, num_clusters: int,  alpha: float = 1.0,
                 tau: float = 0.4, tau_proto: float = 0.2):
        super().__init__(encoder=pretrained_model.encoder, n_hidden=num_hidden, n_proj_hidden=num_proj_hidden, tau=tau)
    
        self.fc_layer1 = pretrained_model.fc_layer1 
        self.fc_layer2 = pretrained_model.fc_layer2
        # self.fuse_fc1 = pretrained_model.fuse_fc1
        # self.fuse_fc2 = pretrained_model.fuse_fc2

        self.alpha: float = alpha  # Parameter for the t-distribution
        # self.kappa: float = kappa  # Parameter for von Mises-Fisher distribution
        # self.tau_proto: float = tau_proto
        self.num_clusters = num_clusters
        # self.cluster_centers = torch.randn(num_clusters, num_hidden, requires_grad=False)
        
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, num_hidden))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
        
        # initial_centers = torch.randn(num_clusters, num_hidden) 
        # self.cluster_centers = torch.nn.Parameter(initial_centers)

        # self.cluster_layer = nn.Linear(num_hidden, num_clusters)
        
        # # Khởi tạo trực giao (Orthogonal) giúp Spectral Clustering hội tụ nhanh hơn
        # nn.init.orthogonal_(self.cluster_layer.weight)

        # self.register_buffer("prototypes", torch.zeros(num_clusters, num_hidden))

    def calculate_q(self, Z: torch.Tensor):
      cluster_centers = self.cluster_centers
      cluster_centers = cluster_centers.to(Z.device)
      # print('Z normalize: ', F.normalize(Z, p = 2, dim = 1))
    #   print('cluster_centers: ', F.normalize(cluster_centers, p = 2, dim = 1))
      dis = torch.cdist(Z, cluster_centers)
    #   dis = torch.cdist(F.normalize(Z, p = 2, dim = 1), F.normalize(cluster_centers, p = 2, dim = 1))
      # Soft assignments (q) using Student's t-distribution
      q = 1.0 / (1.0 + dis**2 / self.alpha)
      q = q ** ((self.alpha + 1.0) / 2.0)
      q = q / (q.sum(dim=1, keepdim=True))  # Normalize to probabilities

      return q 

    # def calculate_q(self, Z: torch.Tensor):
    #     """
    #     Tính soft assignment Q dựa trên vMF kernel (Cosine Similarity).
    #     Công thức: q_ij = exp(kappa * cos(z_i, mu_j)) / sum_k(...)
    #     """
    #     # A. Normalize Z và Tâm cụm về mặt cầu đơn vị (Hypersphere)
    #     # Đây là yêu cầu bắt buộc của vMF/Contrastive Learning [cite: 2650, 2738]
    #     Z_norm = F.normalize(Z, p=2, dim=1)
        
    #     cluster_centers = self.cluster_centers.to(Z.device)
    #     centers_norm = F.normalize(cluster_centers, p=2, dim=1)

    #     # B. Tính Cosine Similarity (Dot Product)
    #     # (N, D) x (D, K) -> (N, K)
    #     logits = torch.matmul(Z_norm, centers_norm.t())

    #     # C. Scale bằng kappa (Concentration parameter)
    #     # Kappa càng lớn -> Phân phối càng nhọn (cụm càng gọn)
    #     logits = logits * self.kappa 

    #     # D. Softmax để ra xác suất
    #     q = F.softmax(logits, dim=1)

    #     return q


    def calculate_p(self, q):
      f = torch.sum(q, dim=0)
      p = (q**2) / (f)
      p =  p / (p.sum(dim=1, keepdim=True))
      return p.detach()


    def clustering_loss_new(self, Z):
      q = self.calculate_q(Z)
      p = self.calculate_p(q)
      q_log = torch.log(q)
      kl = F.kl_div(q_log, p, reduction='batchmean')

      if kl < 0:  
        kl = torch.tensor(0.0, dtype=torch.float64, device=p.device)
      
      # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
      # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
      return kl

      # print(f'Value of q: {q}')
      # print(f'Value of p: {p}')

      # kl = torch.sum(p * (torch.log(p) - torch.log(q)))
      # kl = kl / p.size(0)
      # return F.kl_div(torch.log(q), p, reduction='batchmean')
      # return kl  

    def clustering_loss_new1(self, q, p):
      p_target = p.detach()

    #   confidence_threshold = 0.9

    #   max_prob, _ = torch.max(p_target, dim=1)
    #   confident_mask = max_prob > confidence_threshold
    #   q_confident = q[confident_mask]
    #   p_confident = p_target[confident_mask]
    #   if q_confident.shape[0] == 0:
    #     return torch.tensor(0.0, device=q.device)
      q_log = torch.log(q + 1e-9)
      kl = F.kl_div(q_log, p_target, reduction='batchmean')
    #   q_log_confident = torch.log(q_confident + 1e-9)
    #   kl = F.kl_div(q_log_confident, p_confident, reduction='batchmean')

      kl = torch.nn.functional.relu(kl)
      
      # assert torch.allclose(q.sum(1), torch.ones(q.size(0), dtype=torch.float64, device=q.device), atol=1e-6), "q not normalized"
      # assert torch.allclose(p.sum(1), torch.ones(p.size(0), dtype=torch.float64, device=p.device), atol=1e-6), "p not normalized"
      return kl

    def update_clusters_center(self, Z, q, num_cluster, device):
      # cluster_sizes = torch.sum(q, dim=0)
      # print('Cluster_sizes: ', cluster_sizes)
    #   Z = F.normalize(Z, p=2, dim=1)
      q_trans = q.t()
      clusters_center_updated = torch.matmul(q_trans, Z) / (torch.sum(q, dim=0).unsqueeze(1))
      # clusters_center_updated = torch.matmul(q, Z) / (torch.sum(q, dim=0).unsqueeze(1) + 1e-8)      
    #   self.cluster_centers = F.normalize(clusters_center_updated, p=2, dim=1)
      self.cluster_centers = clusters_center_updated

    def forward_cluster(self, x, adj):
        """
        Trả về Ma trận gán nhãn mềm Y (N x K)
        """
        z = self.forward(x, adj) # Lấy Z từ Encoder + BatchNorm
        
        # Qua lớp linear -> Softmax để ra xác suất
        # (N, Hidden) -> (N, K)
        y_logits = self.cluster_layer(z)
        y_soft = F.softmax(y_logits, dim=1)
        return y_soft, z

    def spectral_loss_sparse(self, Y, edge_index):
        """
        Tính Deep Spectral Clustering Loss (MinCut) trực tiếp trên Edge Index (Sparse).
        Tiết kiệm bộ nhớ, chạy được cho dataset lớn.
        
        Y: [N, K] - Soft Assignment Matrix
        edge_index: [2, E] - Danh sách cạnh (nên là full graph symmetric)
        """
        num_nodes = Y.shape[0]
        device = Y.device

        # 1. Thêm Self-loops (Quan trọng để giữ tính ổn định cho Laplacian)
        edge_index, _ = remove_self_loops(edge_index)
        
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # 2. Tính toán Ma trận kề chuẩn hóa (Normalized Adjacency)
        # Công thức: A_norm = D^-1/2 * A * D^-1/2
        row, col = edge_index_sl
        
        # Tính bậc (Degree) của mỗi node
        deg = degree(row, num_nodes, dtype=Y.dtype)
        
        # Tính D^-1/2
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # Xử lý chia cho 0
        
        # Tính trọng số cho từng cạnh trong edge_index
        # edge_weight_norm = 1 / sqrt(deg_i * deg_j)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 3. Tạo Ma trận Thưa (Sparse Tensor) biểu diễn A_norm
        adj_sparse = torch.sparse_coo_tensor(
            edge_index_sl, 
            edge_weight, 
            (num_nodes, num_nodes)
        ).to(device)
        
        # 4. Tính MinCut Loss = -Trace(Y.T * A_norm * Y) / N
        
        # Bước 4a: Nhân Sparse x Dense: (N, N) x (N, K) -> (N, K)
        # PyTorch hỗ trợ tốt phép nhân này (spmm)
        A_Y = torch.sparse.mm(adj_sparse, Y)
        
        # Bước 4b: Nhân Dense x Dense: (K, N) x (N, K) -> (K, K)
        Y_t_A_Y = torch.matmul(Y.t(), A_Y)
        
        # Bước 4c: Tính Trace (Tổng đường chéo)
        # Chia cho num_nodes để loss không quá lớn với data lớn
        mincut_loss = -torch.trace(Y_t_A_Y) / num_nodes
        
        # 5. Orthogonality Constraint (Tránh suy biến - Collapsing)
        # Yêu cầu: (Y.T * Y) tiến tới I (Ma trận đơn vị)
        # Điều này ép các cụm phải có kích thước xấp xỉ nhau và không chồng lấn.
        
        # Normalize Y theo cột để ổn định
        Y_norm = F.normalize(Y, p=2, dim=0) 
        I = torch.eye(self.num_clusters, device=device)
        
        ortho_loss = torch.norm(torch.matmul(Y_norm.t(), Y_norm) - I)
        
        # Tổng hợp Loss (Lambda thường là 1.0 hoặc 2.0)
        return mincut_loss + 2.0 * ortho_loss
    
    def proto_contrastive_loss(self, z: torch.Tensor, step_proto_assignment=None):
        z_norm = F.normalize(z, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        pos_prototypes = prototypes_norm[step_proto_assignment]
        pos_sim = torch.sum(z_norm * pos_prototypes, dim=1)
        pos_score = torch.exp(pos_sim / self.tau_proto)

        all_sim = torch.mm(z_norm, prototypes_norm.t())
        all_score = torch.exp(all_sim / self.tau_proto)

        loss = -torch.log(pos_score / all_score.sum(dim=1))

        return loss.mean()

class NewModel3(torch.nn.Module):
    def __init__(self, encoder: NewGATEncoderDense, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(NewModel3, self).__init__()
        self.encoder: NewGATEncoderDense = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

    def fuse(self, z: torch.Tensor):
        h = F.elu(self.fuse_fc1(z))
        return self.fuse_fc2(h)  # back to n_hidden

    def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:

        return self.encoder(x, Adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2
    

class UnifiedGATModel(torch.nn.Module):
    def __init__(self, encoder: UnifiedGATEncoder, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(UnifiedGATModel, self).__init__()
        self.encoder: UnifiedGATEncoder = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

        # new fusion head for 4-view fusion
        self.fuse_fc1 = nn.Linear(2 * n_hidden, n_proj_hidden)
        self.fuse_fc2 = nn.Linear(n_proj_hidden, n_hidden)

        self.bn = nn.BatchNorm1d(n_hidden)

    def fuse(self, z: torch.Tensor):
        h = F.elu(self.fuse_fc1(z))
        return self.fuse_fc2(h)  # back to n_hidden

    # def forward(self, x: torch.Tensor, Adj: torch.Tensor) -> torch.Tensor:
    #     return self.encoder(x, Adj)
        # return self.bn(self.encoder(x, Adj))

    def forward(self, x: torch.Tensor, edge_index, edge_weight) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    # Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def sim(self, z1: torch.Tensor, z2: torch.Tensor):
    #     # Tính khoảng cách Euclidean bình phương giữa mọi cặp (Pairwise Distance)
    #     # z1: [N, D], z2: [N, D] -> Output: [N, N]
    #     # Công thức: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        
    #     z1_sq = torch.sum(z1**2, dim=1, keepdim=True)
    #     z2_sq = torch.sum(z2**2, dim=1, keepdim=True)
        
    #     # 2. Tính tích vô hướng (Dot Product)
    #     # prod: [N, N]
    #     prod = torch.mm(z1, z2.t())
        
    #     # 3. Áp dụng hằng đẳng thức: a^2 + b^2 - 2ab
    #     # Broadcasting sẽ tự lo phần kích thước: [N, 1] + [1, N] - [N, N] -> [N, N]
    #     dist_sq = z1_sq + z2_sq.t() - 2 * prod
        
    #     # 4. Quan trọng: Kẹp giá trị (Clamp) để tránh sai số máy tính ra số âm (vd: -0.000001)
    #     dist_sq = torch.clamp(dist_sq, min=1e-6)

    #     d = z1.shape[1]

    #     scale_factor = d ** 0.5
        
    #     # Chuyển thành Similarity: Dùng dấu ÂM
    #     # Càng gần -> khoảng cách càng nhỏ -> sim càng lớn (gần 0)
    #     # Càng xa -> khoảng cách càng lớn -> sim càng nhỏ (âm vô cùng)
    #     return -dist_sq / scale_factor

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size : (i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size : (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret
    
    def contrastive_loss_basic_4views(self,
                                      z1: torch.Tensor,
                                     z2: torch.Tensor,
                                      z3: torch.Tensor,
                                      z4: torch.Tensor,
                                      margin: float = 1.0):
        """
        Basic margin-based contrastive loss trên 4 view:
         - Branch 1: (z1,z2) positive, negatives = z3,z4
         - Branch 2: (z3,z4) positive, negatives = z1,z2
        """
        import torch.nn.functional as F

        # 1) Projection + normalize
        h1 = F.normalize(self.projection(z1), dim=1)
        h2 = F.normalize(self.projection(z2), dim=1)
        h3 = F.normalize(self.projection(z3), dim=1)
        h4 = F.normalize(self.projection(z4), dim=1)

        # 2) Branch 1: positive (h1,h2), negatives h3,h4
        d_pos12  = F.pairwise_distance(h1, h2)
        d_neg13  = F.pairwise_distance(h1, h3)
        d_neg14  = F.pairwise_distance(h1, h4)
        loss1    = (0.5 * d_pos12.pow(2)
                   + 0.5 * F.relu(margin - d_neg13).pow(2)
                   + 0.5 * F.relu(margin - d_neg14).pow(2)).mean()

        # 3) Branch 2: positive (h3,h4), negatives h1,h2
        d_pos34  = F.pairwise_distance(h3, h4)
        d_neg31  = F.pairwise_distance(h3, h1)
        d_neg32  = F.pairwise_distance(h3, h2)
        loss2    = (0.5 * d_pos34.pow(2)
                   + 0.5 * F.relu(margin - d_neg31).pow(2)
                   + 0.5 * F.relu(margin - d_neg32).pow(2)).mean()

        return loss1, loss2