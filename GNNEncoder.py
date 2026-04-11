import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_matrix = nn.Linear(in_features, out_features)

    def forward(self, X, Adj):
        # Aggregate features from neighbors
        aggregated_features = torch.matmul(Adj, X)

        # Apply weighted linear transformation
        output = self.weight_matrix(aggregated_features)

        return output


class SimpleGNNEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.layer1 = SimpleGNNLayer(in_features, hidden_features)
        self.layer2 = SimpleGNNLayer(hidden_features, out_features)


    def forward(self, X, Adj):
        x = self.layer1(X, Adj)
        x = F.relu(x)
        x = self.layer2(x, Adj)

        return x
