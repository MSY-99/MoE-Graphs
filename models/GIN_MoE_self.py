import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GINConv

class SelfLoopMoETop1(nn.Module):
    def __init__(self, in_dim, num_experts=4, capacity_factor=1.0, drop_tokens=False):
        super().__init__()

        self.in_dim = in_dim
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.E = num_experts
        self._aux_mask = None
        
        # ----- Experts ----- #
        self.eps = nn.Parameter(torch.zeros(self.E, 1))

        # ----- Router ----- #
        self.router = nn.Linear(in_dim, self.E, bias=True)

        # ----- Aux loss accumulator ----- #
        self.register_buffer("_aux", torch.tensor(0.0))

        # ----- Initialize parameters ----- #
        nn.init.xavier_uniform_(self.router.weight)


    @property
    def aux(self):
        return self._aux
    
    def forward(self, x):
        N, E = x.size(0), self.E
        device = x.device

        # ----- Router ----- #
        logits = self.router(x)
        self.save_logit = logits.detach().clone()
        probs = F.softmax(logits, dim=-1)

        top1 = torch.argmax(probs, dim=-1)

        # ----- Capacity (per-expert) ----- #
        cap = math.ceil(self.capacity_factor * (N / E))
        idx_by_e = [(top1 == e).nonzero(as_tuple=True)[0] for e in range(E)]

        # ----- Expert execution & gated aggregation ----- #
        y = x.new_zeros(x.shape)

        for e in range(E):
            idx = idx_by_e[e]
            if idx.numel() == 0:
                continue

            if idx.numel() > cap:
                kept, dropped = idx[:cap], idx[cap:]
            else:
                kept, dropped = idx, None

            # α_e = 1 + eps_e
            eps_e = self.eps[e]
            alpha_e = 1.0 + eps_e
            x_e = x[kept] 

            # ----- Straight-through Trick ----- #
            y[kept] = x_e * alpha_e * (1 + probs[kept, e].view(-1, 1) - probs[kept, e].view(-1, 1).detach()) 

            if dropped is not None and not self.drop_tokens:
                y[dropped] = x[dropped]  # α=1

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
        aux = E * torch.sum(importance * load)
        self._aux = aux

        return y
    
    

class GINMoESLConv(GINConv):
    def __init__(self, in_dim, out_dim, num_experts=4, capacity_factor=1.0, drop_tokens=True, **kwargs):
        
        nn_mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
        
        super().__init__(nn=nn_mlp, eps=0.0, train_eps=False, **kwargs)

        self.self_moe = SelfLoopMoETop1(
            in_dim=in_dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            drop_tokens=drop_tokens
        )
        self.register_buffer('_aux0', torch.tensor(0.0))

    @property
    def aux(self):
        return getattr(self.self_moe, 'aux', self._aux0.to(next(self.parameters()).device))

    def forward(self, x, edge_index, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            x_scaled = self.self_moe(x_r)
            out = out + x_scaled

        return self.nn(out)



class NodeGINMoEself(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts=4, capacity_factor=1.0, drop_tokens=False, dropout=0.3):
        super().__init__()

        self.g1 = GINMoESLConv(in_dim, hidden_dim,
                            num_experts=num_experts,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)
        
        self.g2 = GINMoESLConv(hidden_dim, hidden_dim,
                            num_experts=num_experts,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)

        self.cls = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = dropout
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
    
    def forward(self, data, return_aux=False, aux_mask=None):
        x, edge_index = data.x.float(), data.edge_index

        self.g1.self_moe._aux_mask = aux_mask
        self.g2.self_moe._aux_mask = aux_mask

        h1 = self.g1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1_d = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.g2(h1_d, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)

        x = torch.cat([h1, h2], dim=-1)
        out = self.cls(x)

        if return_aux:
            aux = self.g1.aux + self.g2.aux
            self.g1.self_moe._aux_mask = None
            self.g2.self_moe._aux_mask = None

            return out, aux
        
        else:
            self.g1.self_moe._aux_mask = None
            self.g2.self_moe._aux_mask = None

            return out
    