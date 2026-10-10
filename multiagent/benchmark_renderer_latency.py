import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from .model.cache_renderer import load_renderer_checkpoint, pred_to_cache
from .model.benchmark_renderer_optimizations import compare, fused_bound_pack

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'multiagent/outputs_mine'
DESTINATION = SOURCE / 'speed/measurements.json'
LENGTHS = [2**i for i in range(6, 17)]


def visible(path):
    path = Path(path)
    if any(part.startswith('.') for p in (path.absolute(), path.resolve()) for part in p.parts):
        raise ValueError(f'Hidden source/output is not eligible for paper measurements: {path}')
    return path


def current_run_path(recorded):
    run, *rest = Path(recorded).relative_to('outputs').parts
    moves = json.loads(visible(SOURCE / 'swat/legacy_names.json').read_text())['moves']
    return SOURCE / moves.get(run, run) / Path(*rest)


def measure(fn, warmups, repeats):
    for _ in range(warmups):
        value = fn()
        torch.cuda.synchronize()
        del value
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del value
    return dict(status='ok', median_ms=statistics.median(samples), samples_ms=samples,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)


def renderer_carrier(model_path):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    key = 'model.embed_tokens.weight'
    index = json.loads(visible(model_path / 'model.safetensors.index.json').read_text())
    with safe_open(visible(model_path / index['weight_map'][key]), framework='pt', device='cpu') as handle:
        weights = handle.get_tensor(key).to(device='cuda', dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.config = config
    model.layers = torch.nn.ModuleList()
    model.embed_tokens = torch.nn.Embedding.from_pretrained(weights)
    rotary = {'llama': LlamaRotaryEmbedding, 'qwen3': Qwen3RotaryEmbedding}[config.model_type]
    model.rotary_emb = rotary(config, device='cuda')
    return model.eval()


def run_length(args, model, renderer, pack, length):
    ids = torch.randint(10, model.config.vocab_size, (1, length), device='cuda')
    positions = torch.arange(length, device='cuda').unsqueeze(0)
    if renderer is None:
        return measure(lambda: model(input_ids=ids, position_ids=positions, use_cache=True), args.warmups, args.repeats)
    memory = torch.randn(1, length, renderer.sender_flat_dim, device='cuda', dtype=torch.bfloat16)
    virtual = positions.float()
    original_bound = renderer._bound
    renderer._bound = lambda pred, receiver_mask=None: pred

    def forward():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = renderer(model.embed_tokens(ids), positions, memory, virtual)
        cos, sin = model.rotary_emb(pred.new_empty(1, length, model.config.hidden_size), positions)
        keys, values = pack(pred, renderer.stat_key_rms, renderer.stat_value_rms,
                            cos[None, :, None], sin[None, :, None])
        return tuple(zip(keys.unbind(0), values.unbind(0)))

    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            value = forward()
            torch.cuda.synchronize()
            del value
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                value = forward()
                del value
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - start) * 1000

        def replay():
            graph.replay()
            return output

        check = None
        if length in (64, 4096):
            renderer._bound = original_bound
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = renderer(model.embed_tokens(ids), positions, memory, virtual)
            reference = pred_to_cache(model, pred, positions, torch.bfloat16)
            del pred
            replay()
            assert len(reference) == len(output)
            for ref_layer, out_layer in zip(reference, output):
                for ref, out in zip(ref_layer, out_layer):
                    assert ref.shape == out.shape and ref.dtype == out.dtype
                    assert bool(torch.isfinite(ref).all()) and bool(torch.isfinite(out).all())
            check = compare(reference, output)
            assert check['relative_l2'] < 1e-3, check
            del reference
            renderer._bound = lambda pred, receiver_mask=None: pred
        result = measure(replay, args.warmups, args.repeats)
        result.update(setup_ms=setup_ms, execution='fused_cache_cuda_graph')
        if check is not None:
            result['cache_check'] = check
        return result
    finally:
        renderer._bound = original_bound


@torch.inference_mode()
def worker(args):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch._dynamo.config.cache_size_limit = 64
    meta = json.loads(visible(SOURCE / f'kvxfer_bench/arc-c_n500_{args.pair}.json').read_text())
    checkpoint = visible(current_run_path(meta['renderer']) / 'renderer.pt')
    receiver_path = visible(meta['receiver_model'])
    renderer = pack = None
    if args.method == 'reprefill':
        model = AutoModel.from_pretrained(receiver_path, dtype=torch.bfloat16,
                                          attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    else:
        model = renderer_carrier(receiver_path)
        renderer, payload = load_renderer_checkpoint(checkpoint, 'cuda:0')
        del payload
        for module in renderer.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
        pack = torch.compile(fused_bound_pack, fullgraph=True, dynamic=False,
                             options={'emulate_precision_casts': True})
    gc.collect()
    torch.cuda.empty_cache()
    result = dict(pair=args.pair, method=args.method, checkpoint=str(checkpoint),
                  sender_model=meta['sender_model'], receiver_model=str(receiver_path),
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  isolated_process=True, native_max_positions=model.config.max_position_embeddings,
                  execution='eager_sdpa' if renderer is None else 'fused_cache_cuda_graph',
                  resident_parameter_gib=(sum(p.numel()*p.element_size() for p in model.parameters()) +
                    (sum(p.numel()*p.element_size() for p in renderer.parameters()) if renderer is not None else 0))/2**30,
                  warmups=args.warmups, repeats=args.repeats, rows=[])
    for length in args.lengths:
        gc.collect()
        torch.cuda.empty_cache()
        row = dict(tokens=length, beyond_receiver_native_window=length > model.config.max_position_embeddings)
        try:
            row.update(run_length(args, model, renderer, pack, length))
        except Exception as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(exc).lower():
                raise
            row.update(status='oom', error=str(exc))
        result['rows'].append(row)
        print(f"{args.pair} {args.method} {length}: {row.get('median_ms', row['status'])}", file=sys.stderr, flush=True)
    print('RESULT_JSON:' + json.dumps(result), flush=True)


def write_result(path, data):
    path = visible(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=['14E', '15A'], help='Internal isolated worker')
    parser.add_argument('--method', choices=['reprefill', 'renderer'])
    parser.add_argument('--output', type=Path, default=DESTINATION)
    parser.add_argument('--lengths', type=int, nargs='+', default=LENGTHS)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    if bool(args.pair) != bool(args.method):
        parser.error('--pair and --method must be supplied together')
    if args.warmups < 1 or args.repeats < 3 or len(set(args.lengths)) != len(args.lengths) or min(args.lengths) < 1:
        parser.error('Require positive distinct lengths, at least 1 warmup and at least 3 repeats')
    if args.pair:
        worker(args)
        return
    data = dict(schema_version=1, complete=False, started_utc=datetime.now(timezone.utc).isoformat(),
                lengths=args.lengths, warmups=args.warmups, repeats=args.repeats,
                methodology='Batch 1; synthetic equal-length sender/receiver inputs. Receiver: eager SDPA backbone prefill with use_cache=True and no LM head. Renderer: BF16 linear weights, FP32 norms/stat buffers, fused stat bounding/native KV layout/RoPE with precision emulation, fixed-shape CUDA graph replay. Renderer includes embedding lookup and full native KV output. Causal self-attention/full cross-attention, matching historical Speed_diff. Separate sequential processes; same GPU. Synchronized wall-clock medians; setup recorded separately. Excludes sender prefill/flatten/transfer, tokenization, input staging, model loading, graph compilation/capture, and decoding. Graph reuse is assumed. Beyond native context windows, runtime only.',
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), ROOT/'multiagent/model/cache_renderer.py', ROOT/'multiagent/model/benchmark_renderer_optimizations.py')},
                benchmarks={})
    write_result(args.output, data)
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONDONTWRITEBYTECODE='1')
    for pair in ('14E', '15A'):
        data['benchmarks'][pair] = {}
        for method in ('renderer', 'reprefill'):
            command = [sys.executable, '-B', '-m', 'multiagent.benchmark_renderer_latency', '--pair', pair,
                       '--method', method, '--warmups', str(args.warmups), '--repeats', str(args.repeats),
                       '--lengths', *map(str, args.lengths)]
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True, check=True)
            record = next(line[len('RESULT_JSON:'):] for line in reversed(proc.stdout.splitlines()) if line.startswith('RESULT_JSON:'))
            data['benchmarks'][pair][method] = json.loads(record)
            write_result(args.output, data)
    data.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write_result(args.output, data)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
