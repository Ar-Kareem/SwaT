import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from . import cache_renderer as cr

ROOT = Path(__file__).resolve().parents[2]


def measure(fn, repeats=9):
    for _ in range(3):
        out = fn()
        torch.cuda.synchronize()
        del out
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        gpu.append(start.elapsed_time(end))
        del out
    return dict(median_ms=statistics.median(wall), samples_ms=wall,
                gpu_elapsed_ms=statistics.median(gpu))


def shared_rope_cache(model, pred, positions, dtype):
    cos, sin = model.rotary_emb(pred.new_empty(1, positions.shape[1], model.config.hidden_size), positions)
    cos, sin = cos.unsqueeze(1).to(dtype), sin.unsqueeze(1).to(dtype)
    layers = []
    for slot in range(pred.shape[2]):
        key = pred[:, :, slot, 0].permute(0, 2, 1, 3).contiguous().to(dtype)
        value = pred[:, :, slot, 1].permute(0, 2, 1, 3).contiguous().to(dtype)
        half = key.shape[-1] // 2
        rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        layers.append((key * cos + rotated * sin, value))
    return tuple(layers)


def fused_bound(pred, key_rms, value_rms):
    x = pred.float()
    rms = x.square().mean(dim=(1, 5)).sqrt().clamp_min(1e-4)
    scale = torch.stack([key_rms, value_rms], dim=1)[None] / rms
    return (x * scale[:, None, :, :, :, None]).to(pred.dtype)


def fused_bound_pack(pred, key_rms, value_rms, cos, sin):
    bounded = fused_bound(pred, key_rms, value_rms)
    key = bounded[:, :, :, 0].permute(2, 0, 3, 1, 4)
    value = bounded[:, :, :, 1].permute(2, 0, 3, 1, 4).contiguous()
    half = key.shape[-1] // 2
    rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
    key = (key * cos + rotated * sin).contiguous()
    return key, value


def compare(a, b):
    max_abs, diff2, ref2, count, neq = 0., 0., 0., 0, 0
    for aa, bb in zip(a, b):
        for x, y in zip(aa, bb):
            delta = x.float() - y.float()
            max_abs = max(max_abs, delta.abs().max().item())
            diff2 += delta.square().sum().item()
            ref2 += x.float().square().sum().item()
            count += x.numel()
            neq += (x != y).sum().item()
    return dict(max_abs=max_abs, relative_l2=(diff2/max(ref2, 1e-30))**.5,
                changed_fraction=neq/count, bitwise_equal=neq == 0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=['14E', '15A'], required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[1024, 4096, 8192])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / f'multiagent/outputs/renderer_flops_20260919/probes/{args.pair}.json'
    torch.manual_seed(42)
    torch.set_num_threads(4)
    source = ROOT / 'multiagent/outputs_mine'
    meta = json.loads((source / f'kvxfer_bench/arc-c_n500_{args.pair}.json').read_text())
    checkpoint = source / Path(meta['renderer']).relative_to('outputs') / 'renderer.pt'
    renderer, payload = cr.load_renderer_checkpoint(checkpoint, 'cuda:0')
    del payload
    config = AutoConfig.from_pretrained(meta['receiver_model'], local_files_only=True)
    model = torch.nn.Module()
    model.config = config
    model.layers = torch.nn.ModuleList()
    model_path = Path(meta['receiver_model'])
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    key = 'model.embed_tokens.weight'
    with safe_open(model_path / index['weight_map'][key], framework='pt', device='cpu') as sf:
        embedding = sf.get_tensor(key).to(device='cuda', dtype=torch.bfloat16)
    model.embed_tokens = torch.nn.Embedding.from_pretrained(embedding)
    model.rotary_emb = (LlamaRotaryEmbedding if config.model_type == 'llama' else Qwen3RotaryEmbedding)(config, device='cuda')
    original_bound = cr.bound_to_stats
    compiled_bound = torch.compile(fused_bound, fullgraph=True, dynamic=False)
    compiled_pack = torch.compile(fused_bound_pack, fullgraph=True, dynamic=False,
                                  options={'emulate_precision_casts': True})
    result = dict(pair=args.pair, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  methodology='Real renderer/embedding weights, batch 1, synthetic equal-length tokens/memory. Full cross-attention as historical speed benchmark. Three warmups, nine synchronized wall-clock samples; compilation/capture excluded. CUDA events include host launch gaps. Fresh local baseline; no quality eval.', rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.lengths:
        ids = torch.randint(10, config.vocab_size, (1, n), device='cuda')
        positions = torch.arange(n, device='cuda').unsqueeze(0)
        memory = torch.randn(1, n, renderer.sender_flat_dim, device='cuda', dtype=torch.bfloat16)
        virtual = positions.float()
        row = dict(tokens=n, variants={})
        def forward(converter=cr.pred_to_cache):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = renderer(model.embed_tokens(ids), positions, memory, virtual)
            return converter(model, pred, positions, torch.bfloat16)
        for module in renderer.modules():
            if isinstance(module, torch.nn.Linear):
                module.float()
        cr.bound_to_stats = original_bound
        reference = forward()
        row['variants']['baseline'] = measure(forward)
        row['variants']['shared_rope'] = measure(lambda: forward(shared_rope_cache))
        row['variants']['shared_rope']['correctness'] = compare(reference, forward(shared_rope_cache))
        if n == 4096:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                out = forward()
                torch.cuda.synchronize()
            del out
            args.output.with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=45))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bounded_pred = renderer(model.embed_tokens(ids), positions, memory, virtual)
            row['stages'] = dict(cache_conversion=measure(lambda: cr.pred_to_cache(model, bounded_pred, positions, torch.bfloat16)),
                                 bound_only=measure(lambda: original_bound(bounded_pred, renderer.stat_key_rms, renderer.stat_value_rms, False)))
            del bounded_pred
        for module in renderer.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
        row['variants']['precast_linears'] = measure(forward)
        row['variants']['precast_linears']['correctness'] = compare(reference, forward())
        cr.bound_to_stats = lambda pred, k, v, exact, mask=None: compiled_bound(pred, k, v)
        row['variants']['precast_fused_bound_shared_rope'] = measure(lambda: forward(shared_rope_cache))
        row['variants']['precast_fused_bound_shared_rope']['correctness'] = compare(reference, forward(shared_rope_cache))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                tmp = forward(shared_rope_cache)
        torch.cuda.current_stream().wait_stream(stream)
        del tmp
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = forward(shared_rope_cache)
        def replay():
            graph.replay()
            return graph_out
        row['variants']['combined_cuda_graph'] = measure(replay)
        row['variants']['combined_cuda_graph']['correctness'] = compare(reference, replay())
        del graph_out, graph
        cr.bound_to_stats = lambda pred, k, v, exact, mask=None: pred
        def packed_converter(model, pred, positions, dtype):
            cos, sin = model.rotary_emb(pred.new_empty(1, n, config.hidden_size), positions)
            keys, values = compiled_pack(pred, renderer.stat_key_rms, renderer.stat_value_rms,
                                         cos[None, :, None], sin[None, :, None])
            return tuple(zip(keys.unbind(0), values.unbind(0)))
        packed = lambda: forward(packed_converter)
        row['variants']['precast_fused_bound_pack'] = measure(packed)
        row['variants']['precast_fused_bound_pack']['correctness'] = compare(reference, packed())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                tmp = packed()
        torch.cuda.current_stream().wait_stream(stream)
        del tmp
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = packed()
        row['variants']['packed_cuda_graph'] = measure(replay)
        row['variants']['packed_cuda_graph']['correctness'] = compare(reference, replay())
        result['rows'].append(row)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        del reference, graph_out, graph, memory, ids, positions, virtual
        gc.collect()
        torch.cuda.empty_cache()
    cr.bound_to_stats = original_bound


if __name__ == '__main__':
    main()
