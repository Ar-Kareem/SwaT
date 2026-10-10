import torch
from transformers.cache_utils import DynamicCache


class DecodeStep(torch.nn.Module):
    def __init__(self, model, beta_context):
        super().__init__()
        self.model = model
        self.beta_context = beta_context

    def forward(self, input_ids, position_ids, keys, values, betas):
        cache = DynamicCache(config=self.model.config)
        for layer, (key, value) in enumerate(zip(keys, values)):
            cache.update(key, value, layer)
        mask = keys[0].new_zeros((1, 1, 1, keys[0].shape[-2] + 1))
        with self.beta_context(self.model, betas):
            output = self.model(input_ids=input_ids, position_ids=position_ids,
                                cache_position=position_ids.view(-1),
                                attention_mask=mask, past_key_values=cache, use_cache=True)
        return (output.logits[:, -1, :].argmax(-1).view(1, 1), position_ids + 1,
                tuple(layer.keys for layer in cache.layers),
                tuple(layer.values for layer in cache.layers))


def trace_step(model, beta_context, inputs):
    from multiagent.model import attention_matching as am
    original = am._am_eager_attention_forward

    def single_query_attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        return original(module, query, key, value, None, scaling, dropout, **kwargs)

    am._am_eager_attention_forward = single_query_attention
    try:
        with torch.jit.optimized_execution(True):
            return torch.jit.trace(DecodeStep(model, beta_context).eval(), inputs,
                                   check_trace=False, strict=False)
    finally:
        am._am_eager_attention_forward = original


def make_jit_decoder(original_decode, ev, with_prefill=False):
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    from multiagent.model.unaware_comm import cache_layer

    @torch.no_grad()
    def decode(model, tokenizer, cache, betas, receiver_decode_ids, logical_position,
               device, max_new_tokens, min_new_tokens=0, timings=None):
        if min_new_tokens or not str(device).startswith("cuda"):
            return original_decode(model, tokenizer, cache, betas, receiver_decode_ids,
                                   logical_position, device, max_new_tokens, min_new_tokens, timings)
        started = ev.synchronized_time(device)
        keys, values = zip(*(cache_layer(cache, i) for i in range(len(cache.layers))))
        if len(receiver_decode_ids) > 1:
            ids = torch.tensor([receiver_decode_ids[:-1]], dtype=torch.long, device=device)
            positions = torch.arange(logical_position, logical_position + ids.shape[1], device=device).unsqueeze(0)
            if with_prefill and ids.shape[1] <= 256:
                from multiagent.model.am_jit_prefill import trace_prefill
                prefill = getattr(model, "_am_jit_prefill", None)
                if prefill is None:
                    prefill = trace_prefill(model, ev.beta_attention, (ids, positions, keys, values, tuple(betas)))
                    object.__setattr__(model, "_am_jit_prefill", prefill)
                with torch.jit.optimized_execution(True):
                    keys, values = prefill(ids, positions, keys, values, tuple(betas))
                if timings is not None:
                    timings["receiver_jit_prefill"] = True
            else:
                with ev.beta_attention(model, betas):
                    cache = model.model(input_ids=ids, position_ids=positions, past_key_values=cache, use_cache=True).past_key_values
                keys, values = zip(*(cache_layer(cache, i) for i in range(len(cache.layers))))
            logical_position += ids.shape[1]
        prefilled = ev.synchronized_time(device)
        ids = torch.tensor([[receiver_decode_ids[-1]]], dtype=torch.long, device=device)
        positions = torch.tensor([[logical_position]], dtype=torch.long, device=device)
        inputs = (ids, positions, keys, values, tuple(betas))
        engine = getattr(model, "_am_jit_decode", None)
        if engine is None:
            engine = trace_step(model, ev.beta_attention, inputs)
            object.__setattr__(model, "_am_jit_decode", engine)
        stop_ids = ev.eos_ids(tokenizer)
        output_ids, stop_reason = [], "cutoff"
        with torch.jit.optimized_execution(True):
            for _ in range(max_new_tokens):
                ids, positions, keys, values = engine(ids, positions, keys, values, tuple(betas))
                token = int(ids.item())
                if token in stop_ids:
                    stop_reason = "eos"
                    break
                output_ids.append(token)
        if timings is not None:
            timings["receiver_jit"] = True
            timings["receiver_prefill_ms"] = (prefilled - started) * 1000
            timings["receiver_decode_ms"] = (ev.synchronized_time(device) - prefilled) * 1000
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), stop_reason, len(output_ids)

    return decode
