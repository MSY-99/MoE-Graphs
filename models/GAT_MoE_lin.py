import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch import Tensor
from typing import Union, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

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
        self.experts = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=bias) for _ in range(self.E)]
        )
        self.shared = nn.Linear(in_dim, out_dim, bias=bias) if self.shared_expert else None

        # ----- Router ----- #
        self.router = nn.Linear(in_dim, self.E, bias=bias)
        self.shared_router = nn.Linear(in_dim, 1, bias=bias) if self.shared_expert else None

        # ----- Aux loss accumulator ----- #
        self.register_buffer("_aux", torch.tensor(0.0))

        # ----- Initialize parameters ----- #
        self.reset_parameters()


    def reset_parameters(self):
        for lin in self.experts:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None: 
                nn.init.zeros_(lin.bias)
            
        if self.shared is not None:
            nn.init.xavier_uniform_(self.shared.weight)
            if self.shared.bias is not None:
                nn.init.zeros_(self.shared.bias)

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

        topk_index = torch.topk(probs, self.top_k).indices

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

        return y, logits



class GATMoEConv(GATv2Conv):
    def __init__(self, in_dim, out_dim, heads=1, share_weights=False, num_experts=4, top_k=2, shared_expert=False, capacity_factor=1.0, drop_tokens=False, **kwargs):
        super().__init__(in_dim, out_dim, heads=heads, share_weights=share_weights, **kwargs)

        self._moe_modules = []

        def make_moe():
            return MoELinearTopk(
                in_dim, heads * out_dim,
                num_experts=num_experts,
                top_k=top_k,
                shared_expert=shared_expert,
                capacity_factor=capacity_factor,
                drop_tokens=drop_tokens
            )
        
        if hasattr(self, 'lin_l') and hasattr(self, 'lin_r'):
            if share_weights:
                moe = make_moe()
                self.lin_l = self.lin_r = moe
                self._moe_modules = [moe]

            else:
                moe_l, moe_r = make_moe(), make_moe()
                self.lin_l, self.lin_r = moe_l, moe_r
                self._moe_modules = [moe_l, moe_r]

        else:
            moe = make_moe()
            self.lin = moe
            self._moe_modules = [moe]

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, size: Size = None, return_attention_weights: bool = None):
        """
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l, logits = self.lin_l(x)
            x_l = x_l.view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r, _ = self.lin_r(x, logits)
                x_r = x_r.view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l, logits = self.lin_l(x_l)
            x_l = x_l.view(-1, H, C)
            if x_r is not None:
                x_r, _ = self.lin_r(x, logits)
                x_r = x_r.view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)

        else:
            return out
        

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    

    @property
    def aux(self):
        return getattr(self.lin_l, 'aux', torch.tensor(0.0, device=next(self.parameters()).device))



class NodeGATMoElin(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads1=1, heads2=1, num_experts=4, top_k=2, dropout=0.2, shared_expert=False, share_weights=False, capacity_factor=1.0, drop_tokens=False):
        super().__init__()
        assert hidden_dim % heads1 == 0 , "hidden_dim must be divisible by heads1"
        head_dim1 = hidden_dim // heads1

        self.g1 = GATMoEConv(in_dim, head_dim1, heads=heads1,
                            share_weights=share_weights,
                            num_experts=num_experts, top_k=top_k,
                            shared_expert=shared_expert,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)
        
        self.g2 = GATMoEConv(head_dim1, hidden_dim, heads=heads2,
                            share_weights=share_weights,
                            num_experts=num_experts, top_k=top_k,
                            shared_expert=shared_expert,
                            capacity_factor=capacity_factor,
                            drop_tokens=drop_tokens)
        
        self.cls = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data, return_aux=False, aux_mask=None):
        x, edge_index = data.x.float(), data.edge_index

        for m in self.g1._moe_modules:
            m._aux_mask = aux_mask
        for m in self.g2._moe_modules:
            m._aux_mask = aux_mask

        x1 = self.g1(x, edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.g2(x1, edge_index)
        x2 = F.elu(x2)

        out = self.cls(x2)

        if return_aux:
            aux = self.g1.aux + self.g2.aux
            for m in self.g1._moe_modules:
                m._aux_mask = None
            for m in self.g2._moe_modules:
                m._aux_mask = None

            return out, aux
        
        else:
            for m in self.g1._moe_modules:
                m._aux_mask = None
            for m in self.g2._moe_modules:
                m._aux_mask = None

            return out
