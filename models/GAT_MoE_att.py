import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax, add_self_loops

from utils import scatter_add_torch

class AttentionMoETop1(nn.Module):
    def __init__(self, heads, head_dim, num_experts=4, capacity_factor=1.0, drop_tokens=False, negative_slope=0.2):
        super().__init__()

        self.H = heads
        self.C = head_dim
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.negative_slope = negative_slope
        self.E = num_experts
        self._aux_mask = None

        # ----- Experts ----- #
        self.att = nn.Parameter(torch.empty(self.E, self.H, self.C))

        # ----- Router ----- #
        self.router = nn.Linear(self.C, self.E, bias=True)

        # ----- Aux loss accumulator ----- #
        self.register_buffer("_aux", torch.tensor(0.0))

        # ----- Initialize parameters ----- #
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.router.weight)

    @property
    def aux(self):
        return self._aux

    def forward(self, S_ij: torch.Tensor, Z_dst: torch.Tensor, row: torch.Tensor):
        E_edges, H, C = S_ij.shape
        N = Z_dst.size(0)
        E = self.E
        device = S_ij.device

        # ----- Router ----- #
        Z_node = Z_dst.mean(dim=1)
        logits = self.router(Z_node)
        self.save_logit = logits.detach().clone()
        probs  = F.softmax(logits, dim=-1)

        top1 = torch.argmax(probs, dim=-1)

        # ----- Capacity (per expert/node) ----- #
        cap = math.ceil(self.capacity_factor * (N / max(1, E)))
        idx_by_e = [(top1 == e).nonzero(as_tuple=True)[0] for e in range(E)]

        
        # ----- Expert execution & gated aggregation ----- #
        y = S_ij.new_zeros((E_edges, H))

        for e in range(E):
            idx_nodes = idx_by_e[e]
            if idx_nodes.numel() == 0:
                continue

            scores_e_nodes = probs[idx_nodes, e]
            order = torch.argsort(scores_e_nodes, descending=True)
            idx_nodes = idx_nodes[order]

            if idx_nodes.numel() > cap:
                kept_nodes = idx_nodes[:cap]
                dropped_nodes = idx_nodes[cap:]
            else:
                kept_nodes = idx_nodes
                dropped_nodes = None
                
            # node mask (top-1 + capacity)
            node_mask = torch.zeros(N, dtype=torch.bool, device=device)
            node_mask[kept_nodes] = True

            if dropped_nodes is not None and not self.drop_tokens:
                node_mask[dropped_nodes] = True

            edge_mask = node_mask[row]

            if edge_mask.any():
                # calculate the score using the attention of expert e
                scores_e_edge = (S_ij * self.att[e].unsqueeze(0)).sum(dim=-1)
                
                # ----- Straight-through Trick ----- #
                gate_node = probs[:, e]
                gate_edge = gate_node[row]
                scale = 1.0 + gate_edge - gate_edge.detach()
                
                y[edge_mask] += scores_e_edge[edge_mask] * scale[edge_mask].unsqueeze(-1)

        # ----- Aux Mask ----- #
        mask = self._aux_mask
        if mask is not None:
            mask = mask.to(device)
            if mask.dim() > 1:
                mask = mask[:, 0]
        else:
            mask = torch.ones(N, dtype=torch.bool, device=device)

        if mask.any():
            masked_probs = probs[mask]
            importance = masked_probs.mean(dim=0)

            masked_topk = top1[mask]
            load_list = []
            for e in range(E):
                assigned = (masked_topk == e).float() 
                load_list.append(assigned.mean())
            load = torch.stack(load_list)
        else:
            importance = torch.zeros(E, device=device)
            load = torch.zeros(E, device=device)

        # ----- Aux Loss (importance × load) ----- #
        self._aux = E * torch.sum(importance * load)

        return y
    



class GATMoEATTConv(GATv2Conv):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, share_weights=False,
                 dropout=0.2, negative_slope=0.2, add_self_loops=True,
                 num_experts=4, capacity_factor=1.0, drop_tokens=False, **kwargs):
        super().__init__(in_dim, out_dim, heads=heads,concat=concat, share_weights=share_weights, dropout=dropout, add_self_loops=add_self_loops, negative_slope=negative_slope, **kwargs)

        self.att_moe = AttentionMoETop1(
            heads=heads, head_dim=out_dim,
            num_experts=num_experts, 
            capacity_factor=capacity_factor, drop_tokens=drop_tokens,
            negative_slope=negative_slope
        )

    @property
    def aux(self):
        return getattr(self.att_moe, 'aux', torch.tensor(0.0, device=next(self.parameters()).device))


    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels 

        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        if self.add_self_loops and isinstance(x, torch.Tensor):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index[1], edge_index[0]  # dst, src

        # linear transformation
        x_l = self.lin_l(x_dst).view(-1, H, C)
        x_r = x_l if self.share_weights else self.lin_r(x_src).view(-1, H, C)

        S_ij = F.leaky_relu(x_l[row] + x_r[col], negative_slope=self.negative_slope)
        logits = self.att_moe(S_ij, Z_dst=x_l, row=row)

        # softmax (dst)
        alphas = []
        
        for h in range(H):
            a_h = softmax(logits[:, h], row, num_nodes=x_l.size(0))

            if self.dropout > 0 and self.training:
                a_h = F.dropout(a_h, p=self.dropout, training=True)

            alphas.append(a_h)

        alpha = torch.stack(alphas, dim=1)
        out_heads = []

        for h in range(H):
            msg_h = x_r[col, h, :] * alpha[:, h].unsqueeze(-1)
            out_h = scatter_add_torch(row, msg_h, dim_size=x_l.size(0))
            out_heads.append(out_h)

        out = torch.cat(out_heads, dim=-1) if self.concat else torch.stack(out_heads, 1).mean(1)
        
        return out
    



class NodeGATMoEAtt(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads1=1, heads2=1, num_experts=4, dropout=0.2, share_weights=False, capacity_factor=1.0, drop_tokens=False):
        super().__init__()
        assert hidden_dim % heads1 == 0 , "hidden_dim must be divisible by heads1"
        head_dim1 = hidden_dim // heads1

        self.g1 = GATMoEATTConv(in_dim, head_dim1, heads=heads1,
                                concat=True, share_weights=share_weights,
                                dropout=dropout,
                                negative_slope=0.2,
                                add_self_loops=True,
                                num_experts=num_experts,
                                capacity_factor=capacity_factor,
                                drop_tokens=drop_tokens)
        in_dim_g2 = hidden_dim * heads1 if self.g1.concat else hidden_dim

        self.g2 = GATMoEATTConv(in_dim_g2, hidden_dim, heads=heads2,
                                concat=False, share_weights=share_weights,
                                dropout=dropout,
                                negative_slope=0.2,
                                add_self_loops=True,
                                num_experts=num_experts,
                                capacity_factor=capacity_factor,
                                drop_tokens=drop_tokens)
        
        self.cls = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, data, return_aux=False, aux_mask=None):
        x, edge_index = data.x.float(), data.edge_index

        self.g1.att_moe._aux_mask = aux_mask
        self.g2.att_moe._aux_mask = aux_mask

        x1 = self.g1(x, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.g2(x1, edge_index)
        x2 = F.elu(x2)

        out = self.cls(x2)

        if return_aux:
            aux = self.g1.aux + self.g2.aux
            self.g1.att_moe._aux_mask = None
            self.g2.att_moe._aux_mask = None

            return out, aux
        
        else:    
            self.g1.att_moe._aux_mask = None
            self.g2.att_moe._aux_mask = None

            return out