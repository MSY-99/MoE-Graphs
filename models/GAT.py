import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scatter_add_torch
from torch_geometric.utils import add_self_loops, softmax


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.2, negative_slope=0.2, share_weights=False, feat_dropout=0.2, add_self_loops=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.share_weights = share_weights

        if self.concat:
            self.hidden_dim = self.out_dim // self.heads
        else:
            self.hidden_dim = self.out_dim

        self.add_self_loops = add_self_loops
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.feat_dropout = nn.Dropout(feat_dropout) if feat_dropout > 0 else None
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.lin = nn.Linear(self.in_dim, self.hidden_dim*self.heads, bias=True)

        if self.share_weights:
            self.linear_r = self.lin
        else:
            self.linear_r = nn.Linear(self.in_dim, self.hidden_dim*self.heads, bias=True)

        self.attn = nn.Parameter(torch.empty(self.heads, self.hidden_dim))

        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.lin.weight)
        if not self.share_weights:
            nn.init.xavier_uniform_(self.linear_r.weight)

    def forward(self, x, edge_index):
        D_h = self.hidden_dim
        H = self.heads
        N = x.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        row, col = edge_index[1], edge_index[0]  # row=target, col=source

        if self.feat_dropout and self.training:
            x = self.feat_dropout(x)

        Wh_l = self.lin(x)
        Wh_r = self.linear_r(x)   

        Wh_l = Wh_l.view(N, H, D_h)
        Wh_r = Wh_r.view(N, H, D_h)

        m_ij = self.leaky_relu(Wh_l[row] + Wh_r[col])
        logits = (m_ij * self.attn.unsqueeze(0)).sum(dim=-1)

        alpha_list = []
        for h in range(H):
            alpha_h = softmax(logits[:, h], row, num_nodes=N) 
            if self.dropout and self.training:
                alpha_h = self.dropout(alpha_h)
            alpha_list.append(alpha_h)
        alpha = torch.stack(alpha_list, dim=1)

        out_heads = []
        for h in range(H):
            msg_h = Wh_r[col, h, :] * alpha[:, h].unsqueeze(-1)
            out_h = scatter_add_torch(row, msg_h, dim_size=N)
            out_heads.append(out_h)

        if self.concat:
            out = torch.cat(out_heads, dim=-1)
        else:
            out = torch.stack(out_heads, dim=1).mean(dim=1)

        return out


class NodeGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 heads1=1, dropout=0.2, feat_dropout=0.5,
                 negative_slope=0.2):
        super().__init__()
        assert hidden_dim % heads1 == 0, "hidden_dim must be divisible by heads1"

        self.g1 = GATLayer(
            in_dim, hidden_dim,
            heads=heads1, concat=True,
            dropout=dropout, feat_dropout=feat_dropout,
            negative_slope=negative_slope,
            share_weights=False, add_self_loops=True
        )

        self.g2 = GATLayer(
            hidden_dim, hidden_dim,
            heads=1, concat=True,
            dropout=dropout, feat_dropout=feat_dropout,
            negative_slope=negative_slope,
            share_weights=False, add_self_loops=True
        )

        self.cls = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = self.g1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.g2(x, edge_index)
        x = F.elu(x)

        return self.cls(x)