import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GINConv

class MoELinearTopk(nn.Module):
    """
    Top-k Routing MoE Linear layer.

    - Router: Linear(in_dim -> E) followed by Softmax.
    - Assignment: Top-1 routing via argmax, with maximum capacity per expert 
      constrained by a capacity_factor.
    - Overflow Handling: Tokens exceeding capacity are dropped (output set to 0).
    - Auxiliary Loss: Calculated as E * sum_i (importance_i * load_i)
        * importance_i: Mean routing probability for expert i (mean_x p_i(x)).
        * load_i: Fraction of tokens assigned to expert i (mean_x 1[assigned=i]).
    """

    def __init__(self, in_dim, out_dim, num_experts=4, top_k=2, shared_expert=False, bias=True, capacity_factor=1.0, drop_tokens=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shared_expert = shared_expert
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.E = num_experts - int(self.shared_expert)
        self._aux_mask = None
        
        self.top_k = top_k - int(self.shared_expert)
        assert self.top_k < self.E, "top_k is higher than num_experts."

        # ----- Experts ----- #
        experts = []
        for _ in range(self.E):
            layers = [nn.Linear(in_dim, out_dim)]
            layers += [nn.ReLU(), nn.Linear(out_dim, out_dim)]
            experts.append(nn.Sequential(*layers))
        self.experts = nn.ModuleList(experts)

        if self.shared_expert:
            shared_layers = [nn.Linear(in_dim, out_dim)]
            shared_layers += [nn.ReLU(), nn.Linear(out_dim, out_dim)]
            self.shared = nn.Sequential(*shared_layers)
        else:
            self.shared = None
 
        # ----- Router ----- #
        self.router = nn.Linear(in_dim, self.E, bias=bias)
        self.shared_router = nn.Linear(in_dim, 1, bias=bias) if self.shared_expert else None

        # ----- Aux loss accumulator ----- #
        self.register_buffer("_aux", torch.tensor(0.0))

        # ----- Initialize parameters ----- #
        self.reset_parameters()

    
    def reset_parameters(self):
        for mlp in self.experts:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        
        if self.shared is not None:
            for m in self.shared.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.router.weight)

        if self.shared_expert:
            nn.init.xavier_uniform_(self.shared_router.weight)

        
    @property
    def aux(self):
        return self._aux
    
    def forward(self, x, logits=None):
        N, E, D = x.size(0), self.E, self.out_dim
        device = x.device

        # ----- Router ----- #
        if logits == None:
            logits = self.router(x)
        self.save_logit = logits.detach().clone()
        probs = F.softmax(logits, dim=-1)

        topk_index = torch.topk(probs,self.top_k).indices

        # ----- Capacity (per-expert) ----- #
        cap = math.ceil(self.capacity_factor * (N / E) * self.top_k)
        idx_by_e = [torch.nonzero( (topk_index == e).sum(axis=-1).bool() , as_tuple=False).view(-1) for e in range(E)]
        
        # ----- Expert execution & gated aggregation ----- #
        y = x.new_zeros((N, D))

        for e in range(E):
            idx = idx_by_e[e]

            if idx.numel() == 0:
                continue

            scores_e = probs[idx, e]
            order = torch.argsort(scores_e, descending=True)
            idx = idx[order]

            if idx.numel() > cap:
                kept, dropped = idx[:cap], idx[cap:]
            else:
                kept, dropped = idx, None

            # topk softmax weighted-sum
            y_kept = self.experts[e](x[kept]) * probs[kept, e].view(-1,1)
            y[kept] += y_kept
            
            if dropped is not None and not self.drop_tokens:
                y[dropped] += self.experts[e](x[dropped]) * probs[dropped, e].view(-1,1)

        # ----- Shared Expert ----- #
        if self.shared is not None:
            shared_weight = F.sigmoid(self.shared_router(x))
            y = y + (self.shared(x) * shared_weight)

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
            masked_topk = topk_index[mask]
            load_list = []

            for e in range(E):
                assigned = (masked_topk == e).any(dim=-1)
                load_list.append(assigned.float().mean()) 

            load = torch.stack(load_list) / self.top_k

        else:
            importance = torch.zeros(E, device=device)
            load = torch.zeros(E, device=device) / self.top_k
          
        # ----- Aux Loss (importance × load) ----- #
        aux = E * torch.sum(importance * load)
        self._aux = aux

        return y


class GINMoEConv(GINConv):
    def __init__(self, in_dim, out_dim, learn_eps=True, num_experts=4, top_k=2, shared_expert=False, capacity_factor=1.0, drop_tokens=False):

        moe_mlp = MoELinearTopk(
            in_dim, out_dim,
            num_experts=num_experts,
            top_k=top_k,
            shared_expert=shared_expert,
            capacity_factor=capacity_factor,
            drop_tokens=drop_tokens,
            bias=True
        )

        super().__init__(nn=moe_mlp, train_eps=learn_eps)

    @property
    def aux(self):
        return getattr(self.nn, 'aux', torch.tensor(0.0, device=next(self.parameters()).device))



class NodeGINMoEmlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, learn_eps=True, num_experts=4, top_k=2, shared_expert=False, capacity_factor=1.0, drop_tokens=False, dropout=0.3):
        super().__init__()

        self.g1 = GINMoEConv(in_dim, hidden_dim, learn_eps=learn_eps,
                            num_experts=num_experts, top_k=top_k,
                            shared_expert=shared_expert,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)
        
        self.g2 = GINMoEConv(hidden_dim, hidden_dim, learn_eps=learn_eps,
                            num_experts=num_experts, top_k=top_k,
                            shared_expert=shared_expert,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)
    
        self.cls = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = dropout
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

    def forward(self, data, return_aux=False, aux_mask=None):
        x, edge_index = data.x.float(), data.edge_index

        self.g1.nn._aux_mask = aux_mask
        self.g2.nn._aux_mask = aux_mask

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
            self.g1.nn._aux_mask = None
            self.g2.nn._aux_mask = None

            return out, aux
        
        else:
            self.g1.nn._aux_mask = None
            self.g2.nn._aux_mask = None

            return out
    
