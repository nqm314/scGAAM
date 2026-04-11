# This file is implemented based on this repository: https://github.com/levinhcntt/scAGCL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import  add_self_loops, dense_to_sparse, degree, remove_self_loops

from GATEncoder import GATEncoderHybrid
class HybridGATModel(torch.nn.Module):
    def __init__(self, encoder: GATEncoderHybrid, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(HybridGATModel, self).__init__()
        self.encoder: GATEncoderHybrid = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

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


from GNNEncoder import SimpleGNNEncoder
class SimpleGNNModel(torch.nn.Module):
    def __init__(self, encoder: SimpleGNNEncoder, n_hidden: int, n_proj_hidden: int, tau: float = 0.5):
        super(SimpleGNNModel, self).__init__()
        self.encoder: SimpleGNNEncoder = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

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