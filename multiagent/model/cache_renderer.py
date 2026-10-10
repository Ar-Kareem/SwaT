from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import Tensor, nn

from multiagent.model.unaware_comm import _apply_rope, _base_model, cache_layer, kv_layer_indices


def flatten_sender_cache(sender_model, sender_cache, sender_position_ids: Tensor) -> Tensor:
    base = _base_model(sender_model)
    indices = kv_layer_indices(base.config)
    if hasattr(sender_cache, "layers") or hasattr(sender_cache, "key_cache"):
        layers = [(layer_idx, *cache_layer(sender_cache, layer_idx)) for layer_idx in indices]
    else:
        if len(sender_cache) != len(indices):
            raise ValueError(f"sender cache has {len(sender_cache)} layers, expected {len(indices)} kv layers")
        layers = [(layer_idx, key, value) for layer_idx, (key, value) in zip(indices, sender_cache)]
    parts = []
    for layer_idx, key, value in layers:
        deroped = _apply_rope(base, key, layer_idx, sender_position_ids, inverse=True)
        batch, heads, seq_len, head_dim = deroped.shape
        parts.append(deroped.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * head_dim))
        parts.append(value.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * head_dim))
    return torch.cat(parts, dim=-1)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cos_sin(positions: Tensor, dim: int, theta: float, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.to(torch.float32)[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope_apply(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


def _prepare_memory_mask(
    memory_mask: Tensor | None,
    memory_key_limits: Tensor | None,
    memory_len: int,
    dtype: torch.dtype,
    device,
) -> tuple[Tensor | None, Tensor | None]:
    min_value = torch.finfo(dtype).min
    if memory_key_limits is None:
        if memory_mask is None or bool(memory_mask.all()):
            return None, None
        additive = torch.zeros(memory_mask.shape, dtype=dtype, device=device).masked_fill(memory_mask == 0, min_value)
        return additive[:, None, None, :], None
    limits = memory_key_limits.to(device=device)
    allowed = torch.arange(memory_len, device=device)[None, None, :] < limits[:, :, None]
    if memory_mask is not None:
        allowed = allowed & memory_mask.to(device=device, dtype=torch.bool)[:, None, :]
    gate = allowed.any(dim=-1, keepdim=True).to(dtype)
    allowed[..., 0] |= ~allowed.any(dim=-1)
    additive = torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, min_value)
    return additive[:, None, :, :], gate


class _RendererBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_self = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_out = nn.Linear(d_model, d_model, bias=False)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model, bias=False)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def _split(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)

    def forward(
        self,
        hidden: Tensor,
        self_cos: Tensor,
        self_sin: Tensor,
        memory: Tensor | None,
        memory_cos: Tensor | None,
        memory_sin: Tensor | None,
        memory_mask: Tensor | None,
        cross_gate: Tensor | None = None,
        self_mask: Tensor | None = None,
        self_gate: Tensor | None = None,
    ) -> Tensor:
        residual = hidden
        x = self.norm_self(hidden)
        q, k, v = self.self_qkv(x).chunk(3, dim=-1)
        q = _rope_apply(self._split(q), self_cos, self_sin)
        k = _rope_apply(self._split(k), self_cos, self_sin)
        attn = F.scaled_dot_product_attention(q, k, self._split(v), attn_mask=self_mask, is_causal=self_mask is None)
        update = self.self_out(self._merge(attn))
        if self_gate is not None:
            update = update * self_gate
        hidden = residual + update

        if memory is not None:
            residual = hidden
            x = self.norm_cross(hidden)
            q = _rope_apply(self._split(self.cross_q(x)), self_cos, self_sin)
            k, v = self.cross_kv(memory).chunk(2, dim=-1)
            k = _rope_apply(self._split(k), memory_cos, memory_sin)
            attn = F.scaled_dot_product_attention(q, k, self._split(v), attn_mask=memory_mask)
            update = self.cross_out(self._merge(attn))
            if cross_gate is not None:
                update = update * cross_gate
            hidden = residual + update

        return hidden + self.ffn(self.norm_ffn(hidden))


class CacheRenderer(nn.Module):
    def __init__(
        self,
        receiver_hidden: int,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        sender_flat_dim: int,
        d_model: int = 1024,
        n_layers: int = 8,
        n_heads: int = 16,
        ffn_dim: int = 4096,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.receiver_hidden = receiver_hidden
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.sender_flat_dim = sender_flat_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.rope_dim = d_model // n_heads

        self.in_proj = nn.Linear(receiver_hidden, d_model, bias=False)
        self.memory_norm = nn.LayerNorm(sender_flat_dim, elementwise_affine=False)
        self.memory_proj = nn.Linear(sender_flat_dim, d_model, bias=False)
        self.blocks = nn.ModuleList(_RendererBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, receiver_layers * 2 * receiver_kv_heads * receiver_head_dim, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def config_dict(self) -> dict:
        return {
            "receiver_hidden": self.receiver_hidden,
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "sender_flat_dim": self.sender_flat_dim,
            "d_model": self.d_model,
            "n_layers": len(self.blocks),
            "n_heads": self.n_heads,
            "ffn_dim": self.blocks[0].ffn[0].out_features,
            "rope_theta": self.rope_theta,
        }

    def forward(
        self,
        receiver_embeds: Tensor,
        receiver_positions: Tensor,
        memory: Tensor | None = None,
        memory_positions: Tensor | None = None,
        memory_mask: Tensor | None = None,
        memory_key_limits: Tensor | None = None,
    ) -> Tensor:
        hidden = self.in_proj(receiver_embeds)
        self_cos, self_sin = _rope_cos_sin(receiver_positions, self.rope_dim, self.rope_theta, hidden.dtype)
        memory_cos = memory_sin = cross_gate = None
        if memory is not None:
            memory = self.memory_proj(self.memory_norm(memory))
            memory_cos, memory_sin = _rope_cos_sin(memory_positions, self.rope_dim, self.rope_theta, hidden.dtype)
            memory_mask, cross_gate = _prepare_memory_mask(
                memory_mask, memory_key_limits, memory.shape[1], hidden.dtype, memory.device
            )
        for block in self.blocks:
            hidden = block(hidden, self_cos, self_sin, memory, memory_cos, memory_sin, memory_mask, cross_gate)
        out = self.out_proj(self.out_norm(hidden))
        batch, seq, _ = out.shape
        return out.view(batch, seq, self.receiver_layers, 2, self.receiver_kv_heads, self.receiver_head_dim)


def bound_to_stats(pred: Tensor, key_rms: Tensor, value_rms: Tensor, exact_grad: bool, receiver_mask: Tensor | None = None) -> Tensor:
    pred_f = pred.float()
    if receiver_mask is None or bool(receiver_mask.bool().all()):
        rms = pred_f.pow(2).mean(dim=(1, 5)).sqrt().clamp_min(1e-4)
    else:
        valid = receiver_mask.bool()[:, :, None, None, None, None]
        pred_f = pred_f.masked_fill(~valid, 0)
        count = receiver_mask.sum(dim=1).clamp_min(1) * pred.shape[-1]
        rms = (pred_f.pow(2).sum(dim=(1, 5)) / count[:, None, None, None]).clamp_min(1e-8).sqrt()
    if not exact_grad:
        rms = rms.detach()
    target = torch.stack([key_rms, value_rms], dim=1)
    scale = target[None].to(rms.device) / rms
    return (pred_f * scale[:, None, :, :, :, None]).to(pred.dtype)


class CacheRendererV2(nn.Module):

    def __init__(
        self,
        receiver_hidden: int,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        sender_layers: int,
        sender_flat_dim: int,
        d_model: int = 1024,
        n_blocks: int = 8,
        n_heads: int = 16,
        ffn_dim: int = 4096,
        rope_theta: float = 10000.0,
        bound_stats: bool = False,
    ):
        super().__init__()
        if receiver_layers < n_blocks:
            raise ValueError(f"receiver_layers {receiver_layers} must be at least n_blocks {n_blocks}")
        if sender_flat_dim % sender_layers != 0:
            raise ValueError("sender_flat_dim must be a multiple of sender_layers")
        self.receiver_hidden = receiver_hidden
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.sender_layers = sender_layers
        self.sender_flat_dim = sender_flat_dim
        self.sender_layer_dim = sender_flat_dim // sender_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.rope_dim = d_model // n_heads
        self.block_layers = [
            receiver_layers * (b + 1) // n_blocks - receiver_layers * b // n_blocks for b in range(n_blocks)
        ]
        self.layers_per_block = receiver_layers // n_blocks if receiver_layers % n_blocks == 0 else None
        self.sender_slices = [
            (sender_layers * b // n_blocks, sender_layers * (b + 1) // n_blocks) for b in range(n_blocks)
        ]

        self.in_proj = nn.Linear(receiver_hidden, d_model, bias=False)
        self.memory_norms = nn.ModuleList(
            nn.LayerNorm((l1 - l0) * self.sender_layer_dim, elementwise_affine=False) for l0, l1 in self.sender_slices
        )
        self.memory_projs = nn.ModuleList(
            nn.Linear((l1 - l0) * self.sender_layer_dim, d_model, bias=False) for l0, l1 in self.sender_slices
        )
        self.blocks = nn.ModuleList(_RendererBlock(d_model, n_heads, ffn_dim) for _ in range(n_blocks))
        layer_dim = 2 * receiver_kv_heads * receiver_head_dim
        self.out_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_blocks))
        self.out_projs = nn.ModuleList(nn.Linear(d_model, n * layer_dim, bias=False) for n in self.block_layers)
        for proj in self.out_projs:
            nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * n_blocks))

        self.bound_stats = bound_stats
        self.register_buffer("stat_key_rms", torch.ones(receiver_layers, receiver_kv_heads))
        self.register_buffer("stat_value_rms", torch.ones(receiver_layers, receiver_kv_heads))

    def config_dict(self) -> dict:
        return {
            "arch": "v2",
            "receiver_hidden": self.receiver_hidden,
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "sender_layers": self.sender_layers,
            "sender_flat_dim": self.sender_flat_dim,
            "d_model": self.d_model,
            "n_blocks": len(self.blocks),
            "n_heads": self.n_heads,
            "ffn_dim": self.blocks[0].ffn[0].out_features,
            "rope_theta": self.rope_theta,
            "bound_stats": self.bound_stats,
        }

    def load_stat_buffers(self, key_rms: Tensor, value_rms: Tensor):
        self.stat_key_rms.copy_(key_rms.to(self.stat_key_rms.device, self.stat_key_rms.dtype))
        self.stat_value_rms.copy_(value_rms.to(self.stat_value_rms.device, self.stat_value_rms.dtype))

    def forward(
        self,
        receiver_embeds: Tensor,
        receiver_positions: Tensor,
        memory: Tensor | None = None,
        memory_positions: Tensor | None = None,
        memory_mask: Tensor | None = None,
        memory_key_limits: Tensor | None = None,
        receiver_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.in_proj(receiver_embeds)
        self_cos, self_sin = _rope_cos_sin(receiver_positions, self.rope_dim, self.rope_theta, hidden.dtype)
        memory_cos = memory_sin = cross_gate = None
        if memory is not None:
            memory_cos, memory_sin = _rope_cos_sin(memory_positions, self.rope_dim, self.rope_theta, hidden.dtype)
            memory_mask, cross_gate = _prepare_memory_mask(
                memory_mask, memory_key_limits, memory.shape[1], hidden.dtype, memory.device
            )

        batch, seq, _ = receiver_embeds.shape
        self_mask = self_gate = None
        if receiver_mask is not None and not bool(receiver_mask.bool().all()):
            limits = torch.arange(1, seq + 1, device=hidden.device).expand(batch, -1)
            self_mask, self_gate = _prepare_memory_mask(receiver_mask, limits, seq, hidden.dtype, hidden.device)
        per_block = []
        for index, block in enumerate(self.blocks):
            block_memory = None
            if memory is not None:
                l0, l1 = self.sender_slices[index]
                slice_flat = memory[..., l0 * self.sender_layer_dim : l1 * self.sender_layer_dim]
                block_memory = self.memory_projs[index](self.memory_norms[index](slice_flat))
            hidden = block(hidden, self_cos, self_sin, block_memory, memory_cos, memory_sin, memory_mask, cross_gate, self_mask, self_gate)
            out = self.out_projs[index](self.out_norms[index](hidden))
            per_block.append(
                out.view(batch, seq, self.block_layers[index], 2, self.receiver_kv_heads, self.receiver_head_dim)
            )
        pred = torch.cat(per_block, dim=2)
        return self._bound(pred, receiver_mask)

    def _bound(self, pred: Tensor, receiver_mask: Tensor | None = None) -> Tensor:
        if self.bound_stats:
            pred = bound_to_stats(pred, self.stat_key_rms, self.stat_value_rms, getattr(self, "bound_exact_grad", False), receiver_mask)
        if receiver_mask is not None:
            pred = pred.masked_fill(~receiver_mask.bool()[:, :, None, None, None, None], 0)
        return pred


class _PointwiseBlock(nn.Module):

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.norm_fuse = nn.LayerNorm(d_model)
        self.fuse = nn.Linear(d_model, d_model, bias=False)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(self, hidden: Tensor, memory: Tensor | None, gate: Tensor | None) -> Tensor:
        if memory is not None:
            update = self.fuse(self.norm_fuse(memory))
            if gate is not None:
                update = update * gate
            hidden = hidden + update
        return hidden + self.ffn(self.norm_ffn(hidden))


class CacheRendererPointwise(CacheRendererV2):

    def __init__(self, *args, ffn_dim: int = 8192, **kwargs):
        super().__init__(*args, ffn_dim=ffn_dim, **kwargs)
        n_blocks = len(self.blocks)
        self.blocks = nn.ModuleList(_PointwiseBlock(self.d_model, ffn_dim) for _ in range(n_blocks))

    def config_dict(self) -> dict:
        config = super().config_dict()
        config["arch"] = "pointwise"
        config["ffn_dim"] = self.blocks[0].ffn[0].out_features
        return config

    @staticmethod
    def _aligned_index(
        seq: int,
        memory_len: int,
        memory_positions: Tensor | None,
        memory_key_limits: Tensor | None,
        device,
    ) -> tuple[Tensor, Tensor]:
        if memory_key_limits is not None:
            limits = memory_key_limits.to(device=device, dtype=torch.long)
        else:
            receiver_grid = torch.arange(seq, device=device, dtype=memory_positions.dtype)
            limits = (memory_positions.unsqueeze(1) <= receiver_grid.view(1, seq, 1)).sum(dim=-1)
        limits = limits.clamp(max=memory_len)
        gate = (limits > 0).to(torch.float32).unsqueeze(-1)
        index = (limits - 1).clamp(min=0)
        return index, gate

    def forward(
        self,
        receiver_embeds: Tensor,
        receiver_positions: Tensor,
        memory: Tensor | None = None,
        memory_positions: Tensor | None = None,
        memory_mask: Tensor | None = None,
        memory_key_limits: Tensor | None = None,
    ) -> Tensor:
        hidden = self.in_proj(receiver_embeds)
        batch, seq, _ = receiver_embeds.shape
        gathered = gate = None
        if memory is not None:
            index, gate = self._aligned_index(seq, memory.shape[1], memory_positions, memory_key_limits, memory.device)
            gate = gate.to(hidden.dtype)
            gathered = torch.gather(memory, 1, index.unsqueeze(-1).expand(batch, seq, memory.shape[-1]))
        per_block = []
        for block_index, block in enumerate(self.blocks):
            block_memory = None
            if gathered is not None:
                l0, l1 = self.sender_slices[block_index]
                slice_flat = gathered[..., l0 * self.sender_layer_dim : l1 * self.sender_layer_dim]
                block_memory = self.memory_projs[block_index](self.memory_norms[block_index](slice_flat))
            hidden = block(hidden, block_memory, gate)
            out = self.out_projs[block_index](self.out_norms[block_index](hidden))
            per_block.append(
                out.view(batch, seq, self.block_layers[block_index], 2, self.receiver_kv_heads, self.receiver_head_dim)
            )
        pred = torch.cat(per_block, dim=2)
        return self._bound(pred)


class CacheRendererHeadwise(nn.Module):

    def __init__(
        self,
        receiver_layers: int,
        receiver_kv_heads: int,
        receiver_head_dim: int,
        sender_layers: int,
        sender_kv_heads: int,
        sender_head_dim: int,
        hidden_dim: int = 2048,
        bound_stats: bool = False,
    ):
        super().__init__()
        self.receiver_layers = receiver_layers
        self.receiver_kv_heads = receiver_kv_heads
        self.receiver_head_dim = receiver_head_dim
        self.sender_layers = sender_layers
        self.sender_kv_heads = sender_kv_heads
        self.sender_head_dim = sender_head_dim
        self.hidden_dim = hidden_dim
        self.bound_stats = bound_stats
        self.sender_flat_dim = sender_layers * sender_kv_heads * sender_head_dim * 2
        in_dim = 2 * sender_head_dim
        out_dim = 2 * receiver_head_dim
        n = receiver_layers * receiver_kv_heads
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.w1 = nn.Parameter(torch.empty(n, in_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(n, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(n, hidden_dim, out_dim))
        self.b2 = nn.Parameter(torch.zeros(n, out_dim))
        nn.init.normal_(self.w1, std=in_dim ** -0.5)
        nn.init.normal_(self.w2, std=0.02)
        layer_map = [int(round(l * (sender_layers - 1) / max(receiver_layers - 1, 1))) for l in range(receiver_layers)]
        self.register_buffer("layer_map", torch.tensor(layer_map), persistent=False)
        heads = [int(round(h * (sender_kv_heads - 1) / max(receiver_kv_heads - 1, 1))) for h in range(receiver_kv_heads)]
        self.register_buffer("head_map", torch.tensor(heads), persistent=False)
        self.register_buffer("stat_key_rms", torch.ones(receiver_layers, receiver_kv_heads))
        self.register_buffer("stat_value_rms", torch.ones(receiver_layers, receiver_kv_heads))

    def config_dict(self) -> dict:
        return {
            "arch": "headwise",
            "receiver_layers": self.receiver_layers,
            "receiver_kv_heads": self.receiver_kv_heads,
            "receiver_head_dim": self.receiver_head_dim,
            "sender_layers": self.sender_layers,
            "sender_kv_heads": self.sender_kv_heads,
            "sender_head_dim": self.sender_head_dim,
            "hidden_dim": self.hidden_dim,
            "bound_stats": self.bound_stats,
        }

    def load_stat_buffers(self, key_rms: Tensor, value_rms: Tensor):
        self.stat_key_rms.copy_(key_rms.to(self.stat_key_rms.device, self.stat_key_rms.dtype))
        self.stat_value_rms.copy_(value_rms.to(self.stat_value_rms.device, self.stat_value_rms.dtype))

    chunk_tokens: int = 256

    def _mlp(self, x: Tensor) -> Tensor:
        h = F.gelu(torch.baddbmm(self.b1.unsqueeze(1), x, self.w1.to(x.dtype)))
        return torch.baddbmm(self.b2.unsqueeze(1), h, self.w2.to(h.dtype))

    def _render_chunk(self, memory: Tensor, index: Tensor, gate: Tensor) -> Tensor:
        batch, r = index.shape
        gathered = torch.gather(memory, 1, index.unsqueeze(-1).expand(batch, r, memory.shape[-1]))
        x = self.norm(self.matched_inputs(gathered))
        n = self.receiver_layers * self.receiver_kv_heads
        out = self._mlp(x.reshape(batch * r, n, -1).transpose(0, 1))
        out = out.transpose(0, 1).reshape(batch, r, self.receiver_layers, self.receiver_kv_heads, 2, self.receiver_head_dim)
        return out.permute(0, 1, 2, 4, 3, 5) * gate.to(out.dtype).view(batch, r, 1, 1, 1, 1)

    def matched_inputs(self, gathered: Tensor) -> Tensor:
        batch, seq, _ = gathered.shape
        x = gathered.view(batch, seq, self.sender_layers, 2, self.sender_kv_heads, self.sender_head_dim)
        x = x[:, :, self.layer_map]
        x = x[:, :, :, :, self.head_map]
        return x.permute(0, 1, 2, 4, 3, 5).reshape(batch, seq, self.receiver_layers, self.receiver_kv_heads, 2 * self.sender_head_dim)

    def forward(
        self,
        receiver_embeds: Tensor,
        receiver_positions: Tensor,
        memory: Tensor | None = None,
        memory_positions: Tensor | None = None,
        memory_mask: Tensor | None = None,
        memory_key_limits: Tensor | None = None,
    ) -> Tensor:
        if memory is None:
            raise ValueError("headwise renderer has no receiver-embedding path; sender memory is required")
        batch, seq, _ = receiver_embeds.shape
        index, gate = CacheRendererPointwise._aligned_index(seq, memory.shape[1], memory_positions, memory_key_limits, memory.device)
        outs = []
        for start in range(0, seq, self.chunk_tokens):
            idx = index[:, start : start + self.chunk_tokens]
            g = gate[:, start : start + self.chunk_tokens]
            if torch.is_grad_enabled() and (memory.requires_grad or self.w1.requires_grad):
                outs.append(torch.utils.checkpoint.checkpoint(self._render_chunk, memory, idx, g, use_reentrant=False))
            else:
                outs.append(self._render_chunk(memory, idx, g))
        pred = torch.cat(outs, dim=1)
        if self.bound_stats:
            pred = bound_to_stats(pred, self.stat_key_rms, self.stat_value_rms, getattr(self, "bound_exact_grad", False))
        return pred


def pred_to_cache(receiver_model, pred: Tensor, receiver_positions: Tensor, dtype: torch.dtype):
    base = _base_model(receiver_model)
    layers = []
    for slot, layer_idx in enumerate(kv_layer_indices(base.config)):
        key_pre = pred[:, :, slot, 0].permute(0, 2, 1, 3).contiguous().to(dtype)
        value = pred[:, :, slot, 1].permute(0, 2, 1, 3).contiguous().to(dtype)
        key = _apply_rope(base, key_pre, layer_idx, receiver_positions)
        layers.append((key, value))
    return tuple(layers)


def save_renderer_checkpoint(path, renderer: CacheRenderer, extra: dict):
    payload = {
        "renderer_config": renderer.config_dict(),
        "renderer_state_dict": renderer.state_dict(),
        **extra,
    }
    torch.save(payload, path)


def renderer_from_config(config: dict) -> nn.Module:
    config = dict(config)
    arch = config.pop("arch", "v1")
    if arch == "v2":
        return CacheRendererV2(**config)
    if arch == "pointwise":
        return CacheRendererPointwise(**config)
    if arch == "headwise":
        return CacheRendererHeadwise(**config)
    if arch == "dense-latent":
        from multiagent.model.dense_latent_renderer import DenseLatentRenderer
        return DenseLatentRenderer(**config)
    if arch != "v1":
        raise ValueError(f"unknown renderer arch {arch!r}")
    return CacheRenderer(**config)


def load_renderer_checkpoint(path, device: str) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    renderer = renderer_from_config(payload["renderer_config"])
    renderer.load_state_dict(payload["renderer_state_dict"])
    from multiagent.dataset.c2c_token_alignment import checkpoint_alignment
    renderer.token_alignment = checkpoint_alignment(payload)
    renderer = renderer.to(device).eval()
    return renderer, payload
