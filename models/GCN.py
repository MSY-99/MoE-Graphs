import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import scatter_add_torch


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, edge_index):
          x = self.lin(x)
          
          edge_idx, norm_w = gcn_norm(
            edge_index=edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            add_self_loops=True
          )

          row, col = edge_idx[1], edge_idx[0]
          msg = x[col] * norm_w.unsqueeze(-1)
          out = scatter_add_torch(row, msg, dim_size=x.size(0))

          return out


class NodeGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.g1 = GCNLayer(in_dim, hidden_dim)
        self.g2 = GCNLayer(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = self.g1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.g2(x, edge_index)
        x = F.relu(x)
        
        out = self.cls(x)  

        return out
    
