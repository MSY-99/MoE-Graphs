import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scatter_add_torch


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, learn_eps=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.tensor(0.0))

    def forward(self, x, edge_index):
        row, col = edge_index[1], edge_index[0]
        msg = x[col]
        neigh_sum = scatter_add_torch(row, msg, dim_size=x.size(0))

        out_in = (1.0 + self.eps) * x + neigh_sum
        mlp_out = self.mlp(out_in)

        return mlp_out
    

class NodeGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, learn_eps=True):
        super().__init__()
        self.g1 = GINLayer(in_dim, hidden_dim, learn_eps=learn_eps)
        self.g2 = GINLayer(hidden_dim, hidden_dim, learn_eps=learn_eps)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.cls = nn.Linear(hidden_dim*2, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        h1 = self.g1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1_d = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.g2(h1_d, edge_index)    
        h2 = self.bn2(h2)
        h2 = F.relu(h2)

        h = torch.cat([h1, h2], dim=-1)

        out = self.cls(h)

        return out