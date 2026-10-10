from __future__ import annotations

import torch
from torch import nn


class ClosedFormMapper(nn.Module):
    token_alignment = 'native'

    def __init__(self, sender_layers=36, receiver_layers=36, kv_heads=8, head_dim=128,
                 receiver_kv_heads=None, receiver_head_dim=None, k=1, chunk_tokens=256):
        super().__init__()
        if not 1 <= k <= sender_layers:
            raise ValueError(f'k must be in [1, {sender_layers}], got {k}')
        self.sender_layers = sender_layers
        self.receiver_layers = receiver_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.receiver_kv_heads = receiver_kv_heads or kv_heads
        self.receiver_head_dim = receiver_head_dim or head_dim
        self.k = k
        self.chunk_tokens = chunk_tokens
        source_width = kv_heads * head_dim
        target_width = self.receiver_kv_heads * self.receiver_head_dim
        self.register_buffer('source_layers', torch.zeros(receiver_layers, k, dtype=torch.long))
        self.register_buffer('weight', torch.zeros(receiver_layers, 2, k * source_width, target_width))
        self.register_buffer('bias', torch.zeros(receiver_layers, 2, target_width))

    def config_dict(self):
        return dict(arch='closed-form-ridge', sender_layers=self.sender_layers,
                    receiver_layers=self.receiver_layers, kv_heads=self.kv_heads, head_dim=self.head_dim,
                    receiver_kv_heads=self.receiver_kv_heads, receiver_head_dim=self.receiver_head_dim,
                    k=self.k, chunk_tokens=self.chunk_tokens)

    def forward(self, receiver_embeds, receiver_positions, memory,
                memory_positions=None, memory_mask=None, memory_key_limits=None, **kwargs):
        if memory is None:
            raise ValueError('Closed-form baseline requires sender memory')
        batch, source_length, _ = memory.shape
        length = receiver_positions.shape[1]
        source_width = self.kv_heads * self.head_dim
        expected = self.sender_layers * 2 * source_width
        if memory.shape[-1] != expected:
            raise ValueError(f'Expected {expected} source features, got {memory.shape[-1]}')
        if memory_key_limits is None:
            if source_length != length:
                raise ValueError('Without alignment limits the source/target token grids must be identical')
            index = torch.arange(length, device=memory.device)[None].expand(batch, -1)
        else:
            index = (memory_key_limits.long() - 1).clamp(0, source_length - 1)
        source = memory.reshape(batch, source_length, self.sender_layers, 2, source_width)
        source = source[torch.arange(batch, device=memory.device)[:, None], index]
        pred = torch.empty(batch, length, self.receiver_layers, 2, self.receiver_kv_heads * self.receiver_head_dim,
                           device=memory.device, dtype=torch.float32)
        with torch.autocast(device_type=memory.device.type, enabled=False):
            for target in range(self.receiver_layers):
                selected = self.source_layers[target]
                for kind in range(2):
                    for start in range(0, length, self.chunk_tokens):
                        stop = start + self.chunk_tokens
                        x = source[:, start:stop, :, kind].index_select(2, selected).flatten(2).float()
                        pred[:, start:stop, target, kind] = x @ self.weight[target, kind] + self.bias[target, kind]
        return pred.reshape(batch, length, self.receiver_layers, 2, self.receiver_kv_heads, self.receiver_head_dim)


def centered_ridge(x, y, ridge_lambda):
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    return ridge_from_moments(xc.T @ xc, xc.T @ yc, mx, my, ridge_lambda)


def ridge_from_moments(gram, cross, mx, my, ridge_lambda):
    lhs = gram + ridge_lambda * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    factor = torch.linalg.cholesky(lhs)
    weight = torch.cholesky_solve(cross, factor)
    bias = my - mx @ weight
    residual = (lhs @ weight - cross).norm() / cross.norm().clamp_min(1e-30)
    return weight, bias, float(residual)


def same_head_ols_r2(x, y, rcond=1e-10):
    x, y = x.double(), y.double()
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    gram = x.transpose(1, 2) @ x
    cross = x.transpose(1, 2) @ y
    eigenvalues, vectors = torch.linalg.eigh(gram)
    threshold = eigenvalues.amax(-1, keepdim=True) * rcond
    inverse = torch.where(eigenvalues > threshold, eigenvalues.clamp_min(1e-30).reciprocal(), 0)
    weights = (vectors * inverse.unsqueeze(-2)) @ (vectors.transpose(1, 2) @ cross)
    dim = x.shape[-1]
    sst = y.square().sum(1).reshape(x.shape[0], -1, dim).sum(-1)
    sse_channels = y.square().sum(1) - 2 * (weights * cross).sum(1) + (weights * (gram @ weights)).sum(1)
    sse = sse_channels.reshape(x.shape[0], -1, dim).sum(-1).clamp_min(0)
    return torch.where(sst > 1e-20, 1 - sse / sst.clamp_min(1e-20), (sse < 1e-20).double()).T


def all_head_ols_r2(x, y):
    x, y = x.double(), y.double()
    xc, yc = x - x.mean(0), y - y.mean(0)
    gram, cross = xc.T @ xc, xc.T @ yc
    eigenvalues, vectors = torch.linalg.eigh(gram)
    inverse = torch.where(eigenvalues > eigenvalues.max() * 1e-10, eigenvalues.clamp_min(1e-30).reciprocal(), 0)
    weights = (vectors * inverse) @ (vectors.T @ cross)
    sst = yc.square().sum()
    sse = (sst - 2 * (weights * cross).sum() + (weights * (gram @ weights)).sum()).clamp_min(0)
    return float(1 - sse / sst.clamp_min(1e-20))
