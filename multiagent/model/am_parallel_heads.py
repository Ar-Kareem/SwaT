from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import torch

_local = threading.local()


@lru_cache(maxsize=8)
def _pool(workers):
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="am_head")


@torch.no_grad()
def _worker(key, value, queries, target, scaling, config, power, ready):
    from multiagent.model import attention_matching as am
    with torch.cuda.device(key.device):
        stream = getattr(_local, "stream", None)
        if stream is None or stream.device != key.device:
            stream = torch.cuda.Stream(device=key.device)
            _local.stream = stream
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            for tensor in (key, value, queries, power):
                if tensor is not None:
                    tensor.record_stream(stream)
            am._am_head_thread.power = power
            try:
                result = am.compact_head_highest_attention(key, value, queries, target, scaling, config)
            finally:
                del am._am_head_thread.power
            done = torch.cuda.Event()
            done.record(stream)
        return result, done


def resolve_layer(futures):
    outputs = []
    for future in futures:
        result, done = future.result()
        parent = torch.cuda.current_stream(result[0].device)
        parent.wait_event(done)
        for tensor in result:
            tensor.record_stream(parent)
        outputs.append(result)
    return outputs


def submit_layers(model, cache, layer_queries, budget_rows, prefix_len, config, sample):
    from multiagent.model import attention_matching as am
    if not 1 < config.compaction_workers <= 8:
        raise ValueError("Parallel AM compaction_workers must be between 2 and 8")
    specs = {}
    devices = set()
    for i, layer in enumerate(am._base_model(model).layers):
        key, value = am.cache_layer(cache, i)
        if key.shape[2] != prefix_len:
            continue
        queries = am._sample_reference_queries(layer_queries[i].to(key.device), config.query_samples, sample)
        scaling = float(getattr(am._attention_module(layer), "scaling", key.shape[-1] ** -.5))
        specs[i] = [(key[0, h], value[0, h], queries[h], n, scaling, config,
                     torch.randn(n, device=key.device, dtype=torch.float32)
                     if 0 < n < key.shape[2] and config.nnls_iters > 0 else None)
                    for h, n in enumerate(budget_rows[i])]
        devices.add(key.device)
    ready = {}
    for device in devices:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        ready[device] = event
    return {i: [_pool(config.compaction_workers).submit(_worker, *args, ready[args[0].device])
                for args in row] for i, row in specs.items()}


def compact_layer(key, value, queries, budgets, scaling, config):
    from multiagent.model.attention_matching import compact_head_highest_attention
    workers = config.compaction_workers
    if workers < 1 or workers > 8:
        raise ValueError("AM compaction_workers must be between 1 and 8")
    if key.device.type != "cuda" or workers == 1:
        return [compact_head_highest_attention(key[h], value[h], queries[h], n, scaling, config)
                for h, n in enumerate(budgets)]
    parent = torch.cuda.current_stream(key.device)
    powers = [torch.randn(n, device=key.device, dtype=torch.float32)
              if 0 < n < key.shape[1] and config.nnls_iters > 0 else None
              for n in budgets]
    ready = torch.cuda.Event()
    ready.record(parent)
    futures = [_pool(workers).submit(_worker, key[h], value[h], queries[h], n, scaling, config, powers[h], ready)
               for h, n in enumerate(budgets)]
    return resolve_layer(futures)
