from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from multiagent.model.unaware_comm import _apply_rope, _base_model, _kv_shape, cache_layer


class GroupMLP(nn.Module):
    def __init__(self, heads, source_dim, target_dim, expansion=16, dtype=torch.float32):
        super().__init__()
        hidden = expansion * source_dim
        self.w1 = nn.Parameter(torch.empty(heads, source_dim, hidden, dtype=dtype))
        self.b1 = nn.Parameter(torch.empty(heads, 1, hidden, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(heads, hidden, target_dim, dtype=dtype))
        self.b2 = nn.Parameter(torch.empty(heads, 1, target_dim, dtype=dtype))
        for p, fan in [(self.w1, source_dim), (self.b1, source_dim), (self.w2, hidden), (self.b2, hidden)]:
            nn.init.uniform_(p, -1 / math.sqrt(fan), 1 / math.sqrt(fan))

    def forward(self, x):
        return torch.bmm(torch.nn.functional.gelu(torch.bmm(x, self.w1) + self.b1), self.w2) + self.b2


class DenseLatentAdapter(nn.Module):
    def __init__(self, sender, receiver, expansion=16, dtype=torch.float32):
        super().__init__()
        self.__dict__['sender_base'] = _base_model(sender)
        self.__dict__['receiver_base'] = _base_model(receiver)
        sl, sh, sd = _kv_shape(sender)
        rl, rh, rd = _kv_shape(receiver)
        self.source_layers = [round(l * (sl - 1) / max(rl - 1, 1)) for l in range(rl)]
        self.register_buffer('source_heads', torch.tensor([round(h * (sh - 1) / max(rh - 1, 1)) for h in range(rh)]))
        self.keys = nn.ModuleList([GroupMLP(rh, sd, rd, expansion, dtype) for _ in range(rl)])
        self.values = nn.ModuleList([GroupMLP(rh, sd, rd, expansion, dtype) for _ in range(rl)])
        self.gate_logits = nn.Parameter(torch.zeros(rl, rh, dtype=dtype))

    def layer(self, cache, layer, source_indices, valid=None, checkpointed=True):
        source_layer = self.source_layers[layer]
        k, v = cache_layer(cache, source_layer)
        if k.shape[0] != 1:
            raise ValueError('Dense baseline currently requires batch size one')
        k = _apply_rope(self.sender_base, k, source_layer, inverse=True)
        device = self.gate_logits.device
        heads = self.source_heads.to(k.device)
        indices = source_indices.to(k.device)
        k = k[0].index_select(0, heads).index_select(1, indices).to(device)
        v = v[0].index_select(0, heads).index_select(1, indices).to(device)
        def project(k, v):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                gate = self.gate_logits[layer].sigmoid()[:, None, None]
                pk = self.keys[layer](k.to(self.keys[layer].w1.dtype)) * gate
                pv = self.values[layer](v.to(self.values[layer].w1.dtype)) * gate
            return pk.to(k.dtype).unsqueeze(0), pv.to(v.dtype).unsqueeze(0)
        if checkpointed and torch.is_grad_enabled():
            pk, pv = checkpoint(project, k, v, use_reentrant=False)
        else:
            pk, pv = project(k, v)
        if valid is not None:
            mask = valid.to(device=device, dtype=pk.dtype)[None, None, :, None]
            pk, pv = pk * mask, pv * mask
        pk = _apply_rope(self.receiver_base, pk, layer)
        return pk, pv

    def forward(self, cache, source_indices, valid=None):
        return tuple(self.layer(cache, l, source_indices, valid) for l in range(len(self.keys)))
