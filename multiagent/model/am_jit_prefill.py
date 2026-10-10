import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


def prefill_attention(module, query, key, value, attention_mask, scaling,
                      dropout=0., **kwargs):
    groups = module.num_key_value_groups
    keys = key.repeat_interleave(groups, dim=1)
    values = value.repeat_interleave(groups, dim=1)
    qlen, klen = query.shape[-2], key.shape[-2]
    visible = (torch.arange(klen, device=query.device).view(1, -1)
               <= torch.arange(qlen, device=query.device).view(-1, 1) + klen - qlen)
    mask = torch.zeros_like(visible, dtype=query.dtype).masked_fill(
        ~visible, torch.finfo(query.dtype).min)
    beta = module._am_beta.to(device=query.device, dtype=query.dtype)
    beta = beta.repeat_interleave(groups, dim=0)
    beta = F.pad(beta, (0, klen - beta.shape[-1]))
    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    weights = weights + mask.view(1, 1, qlen, klen)
    weights = weights + beta.view(1, beta.shape[0], 1, beta.shape[1])
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, values).transpose(1, 2).contiguous(), None


class PrefillStep(torch.nn.Module):
    def __init__(self, model, beta_context):
        super().__init__()
        self.model = model
        self.beta_context = beta_context

    def forward(self, input_ids, position_ids, keys, values, betas):
        cache = DynamicCache(config=self.model.config)
        for layer, (key, value) in enumerate(zip(keys, values)):
            cache.update(key, value, layer)
        mask = keys[0].new_zeros((1, 1, input_ids.shape[1], keys[0].shape[-2] + input_ids.shape[1]))
        with self.beta_context(self.model, betas):
            self.model.model(input_ids=input_ids, position_ids=position_ids,
                             cache_position=position_ids.view(-1),
                             attention_mask=mask, past_key_values=cache, use_cache=True)
        return (tuple(layer.keys for layer in cache.layers),
                tuple(layer.values for layer in cache.layers))


def trace_prefill(model, beta_context, inputs):
    from multiagent.model import attention_matching as am
    original = am._am_eager_attention_forward
    am._am_eager_attention_forward = prefill_attention
    try:
        with torch.jit.optimized_execution(True):
            return torch.jit.trace(PrefillStep(model, beta_context).eval(), inputs,
                                   check_trace=False, strict=False)
    finally:
        am._am_eager_attention_forward = original
