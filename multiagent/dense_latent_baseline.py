from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from multiagent.dataset.c2c import apply_chat_template, turn_end_ids
from multiagent.dataset.c2c_cross_family import encode_context_pair
from multiagent.dataset.c2c_registry import load_c2c_task, load_c2c_dataset_spec
from multiagent.model.dense_latent import DenseLatentAdapter
from multiagent.model.unaware_comm import cache_layer, full_sender_cache, tuple_to_dynamic_cache, _base_model, kv_layer_indices
from multiagent.model.vanilla_loader import resolve_model_path, load_causal_model

TRAIN = ['gsm8k:train', 'math-algebra:train', 'arc-c:train']
EVAL = ['gsm8k:test', 'math-500:test', 'arc-c:test', 'mmlu-redux:test', 'medqa:test', 'openbookqa:test']
METHOD = 'dense_latent_paper_v2_fp32_master'


def normalized_eval_config(config):
    return {'receiver_prompt_style': 'chat', 'trace_manifest': None, **config}


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_traces(path, receiver_model):
    pool = list(load_c2c_dataset_spec(path))
    expected = {e.id: e for spec in TRAIN for e in load_c2c_dataset_spec(spec)}
    if len(pool) != len(expected) or {e.id for e in pool} != set(expected):
        raise ValueError('receiver trace pool must contain each training example exactly once')
    for e in pool:
        original = expected[e.id]
        if e.metadata.get('receiver_model') != receiver_model or e.metadata.get('trace_mode') != 'receiver_self_guided':
            raise ValueError(f'incorrect trace provenance: {e.id}')
        for key in ('question', 'choices', 'labels'):
            if e.metadata.get(key) != original.metadata.get(key):
                raise ValueError(f'trace/source mismatch for {key}: {e.id}')
        if e.metadata.get('gold_answer') != original.metadata['answer'] or not e.target_text.strip():
            raise ValueError(f'empty trace or incorrect gold answer: {e.id}')
    return pool


@contextmanager
def checkpoint_receiver(receiver):
    from torch.utils.checkpoint import checkpoint
    originals = []
    def wrap(original, layer_index):
        def forward(hidden_states, *positional, **kwargs):
            if positional or kwargs.get('use_cache', False):
                raise ValueError('checkpoint receiver requires keyword args and use_cache=False')
            kwargs = dict(kwargs)
            cache = kwargs.pop('past_key_values')
            k, v = cache_layer(cache, layer_index)
            def compute(h, k, v):
                fresh = DynamicCache()
                fresh.update(k, v, layer_index)
                return original(h, past_key_values=fresh, **kwargs)
            return checkpoint(compute, hidden_states, k, v, use_reentrant=False)
        return forward
    try:
        base = _base_model(receiver)
        for index in kv_layer_indices(base.config):
            layer = base.layers[index]
            originals.append((layer, layer.forward))
            layer.forward = wrap(layer.forward, index)
        yield
    finally:
        for layer, original in originals:
            layer.forward = original


def source_messages(example):
    m = example.metadata
    if example.dataset in {'gsm8k', 'math-algebra', 'math-500'} and 'question' in m:
        ending = 'End with Answer: <number>.' if example.dataset == 'gsm8k' else r'End with the answer in \boxed{}.'
        text = f"{m['question']}\n\nSolve step by step. {ending}"
    elif 'question' in m and 'choices' in m and 'labels' in m:
        choices = '\n'.join(f'{label}. {choice}' for label, choice in zip(m['labels'], m['choices']))
        text = f"{m['question']}\n{choices}\n\nExplain step by step. End with Answer: <letter>."
    else:
        return example.sender_messages
    return [{'role': 'user', 'content': text}]


def encode(sender_tokenizer, receiver_tokenizer, messages, budget):
    row = encode_context_pair(sender_tokenizer, receiver_tokenizer, messages, budget, 'interp')
    if row['sender_ids'] == row['receiver_ids']:
        indices = torch.arange(len(row['receiver_ids']))
        valid = torch.ones_like(indices, dtype=torch.bool)
    else:
        indices = torch.tensor(row['cross_causal_limits'], dtype=torch.long) - 1
        valid = indices >= 0
        indices = indices.clamp(0, len(row['sender_ids']) - 1)
    return row, indices, valid


def receiver_prompt(tokenizer, messages, aware, style='chat'):
    if style == 'base-completion':
        text = '\nProblem: ' + '\n'.join(m['content'] for m in messages) if aware else ''
        text += '\nSolution:'
        return tokenizer(text, add_special_tokens=False)['input_ids']
    if not aware:
        messages = [{'role': 'user', 'content': 'Solve the task represented by the preceding memory. Explain step by step and finish with the final answer in the requested format.'}]
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)['input_ids']


def ids_tensor(ids, device):
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def prefill(model, ids, device):
    return model.model(input_ids=ids_tensor(ids, device), past_key_values=full_sender_cache(), use_cache=True).past_key_values


def load_pair(args):
    sp, rp = resolve_model_path(args.sender_model), resolve_model_path(args.receiver_model)
    st, rt = AutoTokenizer.from_pretrained(sp), AutoTokenizer.from_pretrained(rp)
    for tok in [st, rt]:
        tok.pad_token = tok.pad_token or tok.eos_token
    sender = load_causal_model(sp, torch.bfloat16, args.sender_device)
    receiver = load_causal_model(rp, torch.bfloat16, args.receiver_device)
    sender.requires_grad_(False)
    receiver.requires_grad_(False)
    adapter = DenseLatentAdapter(sender, receiver).to(args.sender_device)
    checkpoint_path = args.checkpoint
    if getattr(args, 'resume', False) and (Path(args.output) / 'latest.pt').exists():
        checkpoint_path = Path(args.output) / 'latest.pt'
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state.get('method') != METHOD:
            raise ValueError('checkpoint method/version mismatch')
        for key in ['sender_model', 'receiver_model']:
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'checkpoint {key} differs from requested model')
        if args.mode == 'eval' and state['args'].get('receiver_prompt_style', 'chat') != args.receiver_prompt_style:
            raise ValueError('evaluation prompt style differs from phase-II training')
        adapter.load_state_dict(state['adapter'])
    print(json.dumps({'trainable_parameters': sum(p.numel() for p in adapter.parameters()), 'args': vars(args)}), flush=True)
    return st, rt, sender, receiver, adapter


def reconstruction(adapter, sender, receiver, row, indices, valid, args, backward=True):
    sc = prefill(sender, row['sender_ids'], args.sender_device)
    rc = prefill(receiver, row['receiver_ids'], args.receiver_device)
    total = 0.0
    for l in range(len(adapter.keys)):
        pk, pv = adapter.layer(sc, l, indices, valid, checkpointed=False)
        tk, tv = cache_layer(rc, l)
        tk, tv = tk.to(pk.device), tv.to(pv.device)
        mask = valid.to(pk.device)[None, None, :, None]
        denom = valid.sum().clamp_min(1) * pk.shape[1] * pk.shape[-1]
        loss = (((pk.float() - tk.float()).square() * mask).sum() + ((pv.float() - tv.float()).square() * mask).sum()) / (2 * denom * len(adapter.keys))
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite reconstruction loss')
        if backward:
            loss.backward()
        total += loss.detach().item()
    return total


def generation_loss(adapter, sender, receiver, row, indices, valid, prompt, target, args):
    sc = prefill(sender, row['sender_ids'], args.sender_device)
    kv = tuple((k.to(args.receiver_device), v.to(args.receiver_device)) for k, v in adapter(sc, indices, valid))
    cache = tuple_to_dynamic_cache(kv, receiver.config)
    sequence = prompt + target
    if not target or not prompt:
        raise ValueError('empty target or receiver prompt')
    inputs = ids_tensor(sequence[:-1], args.receiver_device)
    labels = ids_tensor([-100] * (len(prompt) - 1) + target, args.receiver_device)
    mask = torch.cat([valid[None].to(args.receiver_device), torch.ones_like(inputs, dtype=torch.bool)], dim=1)
    with checkpoint_receiver(receiver):
        output = receiver.model(input_ids=inputs, attention_mask=mask, past_key_values=cache, use_cache=False)
    hidden = output.last_hidden_state[:, len(prompt) - 1:]
    from torch.utils.checkpoint import checkpoint
    losses = []
    targets = labels[:, len(prompt) - 1:]
    for start in range(0, hidden.shape[1], 64):
        def ce(h, y):
            return F.cross_entropy(receiver.lm_head(h).float().reshape(-1, receiver.config.vocab_size), y.reshape(-1), reduction='sum')
        losses.append(checkpoint(ce, hidden[:, start:start + 64], targets[:, start:start + 64], use_reentrant=False))
    return torch.stack(losses).sum() / len(target)


def save(adapter, args, name, step, optimizer=None, rng=None, order=None):
    path = Path(args.output) / name
    temp = path.with_suffix('.tmp')
    torch.save({'adapter': adapter.state_dict(), 'args': vars(args), 'step': step, 'method': METHOD,
                'optimizer': None if optimizer is None else optimizer.state_dict(),
                'rng': None if rng is None else rng.getstate(), 'order': order,
                'torch_rng': torch.get_rng_state()}, temp)
    temp.replace(path)


def preflight(args):
    st, rt, sender, receiver, adapter = load_pair(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0, foreach=False)
    before = adapter.keys[0].w1.detach().flatten()[:4096].clone()
    row = dict(sender_ids=[st.encode('test', add_special_tokens=False)[0]] * args.max_length,
               receiver_ids=[rt.encode('test', add_special_tokens=False)[0]] * args.max_length)
    indices = torch.arange(args.max_length)
    valid = torch.ones(args.max_length, dtype=torch.bool)
    rec = reconstruction(adapter, sender, receiver, row, indices, valid, args)
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss = generation_loss(adapter, sender, receiver, row, indices, valid,
                           row['receiver_ids'], [rt.eos_token_id] * args.max_trace_length, args)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(loss) or float(norm) == 0:
        raise FloatingPointError('preflight CE/gradient failure')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if torch.equal(before, adapter.keys[0].w1.detach().flatten()[:4096]):
        raise RuntimeError('preflight optimizer did not update adapter weights')
    if any(p.grad is not None for model in (sender, receiver) for p in model.parameters()):
        raise RuntimeError('preflight unexpectedly differentiated a frozen language model')
    example = load_c2c_task('gsm8k:train').dataset[0]
    messages = source_messages(example)
    row, indices, valid = encode(st, rt, messages, args.max_length)
    with torch.no_grad():
        sc = prefill(sender, row['sender_ids'], args.sender_device)
        kv = tuple((k.to(args.receiver_device), v.to(args.receiver_device)) for k, v in adapter(sc, indices, valid))
        text, tokens = decode(receiver, rt, receiver_prompt(rt, messages, False, args.receiver_prompt_style),
                              tuple_to_dynamic_cache(kv, receiver.config), valid, args.receiver_device, 4)
    result = dict(passed=True, reconstruction=rec, ce=float(loss), grad_norm=float(norm),
                  args=vars(args), trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                  decoded_tokens=len(tokens), peak_allocated_gib={str(i): torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())})
    (output / 'passed.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def train(args):
    st, rt, sender, receiver, adapter = load_pair(args)
    if args.mode == 'phase2':
        if not args.checkpoint or not args.traces:
            raise ValueError('phase2 requires reconstruction checkpoint and receiver-self traces')
        pool = validate_traces(args.traces, args.receiver_model)
    else:
        pool = [e for spec in TRAIN for e in load_c2c_dataset_spec(spec)]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0, foreach=False)
    rng = random.Random(args.seed)
    order = list(range(len(pool)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = dict(vars(args), method=METHOD, training_examples=len(pool),
                  torch_version=torch.__version__,
                  sender_config=sender.config.to_dict(), receiver_config=receiver.config.to_dict(),
                  trace_sha256=file_digest(args.traces) if args.traces else None,
                  source_sha256={str(p): file_digest(p) for p in [Path(__file__), Path(__file__).parent / 'model/dense_latent.py']})
    recipe = json.loads(json.dumps(recipe))
    recipe_path = output / 'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('training recipe or source code changed; use a new output directory')
    recipe_path.write_text(json.dumps(recipe, indent=2))
    start_step = 0
    if args.resume and (output / 'latest.pt').exists():
        state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if state['args'] != vars(args):
            raise ValueError('resume configuration mismatch')
        optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['rng'])
        torch.set_rng_state(state['torch_rng'])
        order, start_step = state['order'], state['step']
        del state
    elif (output / 'metrics.jsonl').exists():
        raise ValueError('existing metrics without a resumable checkpoint; use a new output directory')
    started = time.time()
    with (output / 'metrics.jsonl').open('a', buffering=1) as log:
        for step in range(start_step + 1, args.steps + 1):
            offset = (step - 1) % len(order)
            if offset == 0:
                rng.shuffle(order)
            example = pool[order[offset]]
            messages = source_messages(example)
            row, indices, valid = encode(st, rt, messages, args.max_length)
            optimizer.zero_grad(set_to_none=True)
            aware = None
            if args.mode == 'phase1':
                value = reconstruction(adapter, sender, receiver, row, indices, valid, args)
            else:
                aware = rng.random() < 0.5
                prompt = receiver_prompt(rt, messages, aware, args.receiver_prompt_style)[:args.max_length]
                target = rt(example.target_text, add_special_tokens=False)['input_ids']
                target = (target + [rt.eos_token_id])[:args.max_trace_length]
                loss = generation_loss(adapter, sender, receiver, row, indices, valid, prompt, target, args)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite CE')
                loss.backward()
                value = loss.detach().item()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, loss=value, grad_norm=float(norm), aware=aware, example=example.id, elapsed=time.time() - started)
            log.write(json.dumps(record) + '\n')
            if step == 1 or step % 25 == 0:
                print(json.dumps(record), flush=True)
            if args.save_every > 0 and step % args.save_every == 0:
                save(adapter, args, 'latest.pt', step, optimizer, rng, order)
        save(adapter, args, 'latest.pt', args.steps, optimizer, rng, order)
        save(adapter, args, 'final.pt', args.steps)


@torch.no_grad()
def decode(receiver, tokenizer, prompt, cache, valid, device, budget, stop_at_problem_boundary=False):
    inputs = ids_tensor(prompt, device)
    n = 0 if cache is None else cache.get_seq_length()
    mask = torch.cat([valid[None].to(device), torch.ones_like(inputs, dtype=torch.bool)], 1) if n else torch.ones_like(inputs)
    eos = turn_end_ids(tokenizer, receiver.generation_config)
    generated = []
    for _ in range(budget):
        out = receiver(input_ids=inputs, past_key_values=cache, attention_mask=mask, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1].argmax(-1)
        generated.append(int(token))
        cache = out.past_key_values
        if int(token) in eos:
            break
        if stop_at_problem_boundary:
            import re
            tail = tokenizer.decode(generated[-32:], skip_special_tokens=True)
            if re.search(r'\n\s*(?:Problem|Question):', tail):
                break
        inputs = token[:, None]
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_at_problem_boundary:
        from multiagent.dataset.dense_base_traces import extract
        text = extract(text)
    return text, generated


@torch.no_grad()
def evaluate(args):
    st, rt, sender, receiver, adapter = load_pair(args)
    adapter.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = file_digest(args.checkpoint)
    for spec in args.tasks:
        task = load_c2c_task(spec)
        order = list(range(len(task.dataset)))
        if not getattr(task.dataset, 'preserve_order', False):
            random.Random(f'{args.seed}:{spec}').shuffle(order)
        if args.limit:
            order = order[:args.limit]
        for mode in args.eval_modes:
            path = output / (spec.replace(':', '_') + '_' + mode + '.jsonl')
            manifest = path.with_suffix('.recipe.json')
            config = dict(vars(args), checkpoint_sha256=checkpoint_sha256, method=METHOD)
            if manifest.exists():
                if normalized_eval_config(json.loads(manifest.read_text())) != normalized_eval_config(config):
                    raise ValueError(f'eval configuration changed for {path}')
            else:
                manifest.write_text(json.dumps(config, indent=2))
            done = {json.loads(line)['id'] for line in path.read_text().splitlines()} if path.exists() else set()
            with path.open('a', buffering=1) as log:
                for index in order:
                    example = task.dataset[index]
                    if example.id in done:
                        continue
                    messages = source_messages(example)
                    row, indices, valid = encode(st, rt, messages, args.max_length)
                    prompt = receiver_prompt(rt, messages, mode != 'unaware', args.receiver_prompt_style)[:args.max_length]
                    cache = None
                    if mode != 'receiver':
                        sc = prefill(sender, row['sender_ids'], args.sender_device)
                        kv = tuple((k.to(args.receiver_device), v.to(args.receiver_device)) for k, v in adapter(sc, indices, valid))
                        cache = tuple_to_dynamic_cache(kv, receiver.config)
                    text, tokens = decode(receiver, rt, prompt, cache, valid, args.receiver_device, args.max_new_tokens,
                                          stop_at_problem_boundary=args.receiver_prompt_style == 'base-completion')
                    grade = task.grade(text, example)
                    stopped = bool(tokens and tokens[-1] in turn_end_ids(rt, receiver.generation_config))
                    log.write(json.dumps(dict(id=example.id, index=index, prediction=text, grade=grade, generated_tokens=len(tokens),
                                              stop_reason='eos' if stopped else 'cutoff',
                                              source_tokens=0 if mode == 'receiver' else len(row['sender_ids']),
                                              cache_tokens=0 if mode == 'receiver' else len(row['receiver_ids']),
                                              receiver_prompt_tokens=len(prompt))) + '\n')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            scores = [float(r['grade'] == 'correct') if isinstance(r['grade'], str) else float(r['grade'].get('score', r['grade'].get('f1', 0))) for r in records]
            if len(scores) != len(order) or {r['id'] for r in records} != {task.dataset[i].id for i in order}:
                raise ValueError(f'eval sample coverage mismatch: {path}')
            summary = dict(task=spec, mode=mode, n=len(scores), score=sum(scores) / len(scores),
                           cutoff_count=sum(r['stop_reason'] == 'cutoff' for r in records),
                           checkpoint_sha256=checkpoint_sha256, method=METHOD)
            path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'phase1', 'phase2', 'eval'])
    p.add_argument('--sender-model', required=True)
    p.add_argument('--receiver-model', required=True)
    p.add_argument('--sender-device', default='cuda:0')
    p.add_argument('--receiver-device', default='cuda:1')
    p.add_argument('--checkpoint')
    p.add_argument('--resume', action='store_true', help='Resume optimizer and sampler from output/latest.pt if present')
    p.add_argument('--traces')
    p.add_argument('--trace-manifest', help='Explicit accepted/rejected training partition for filtered base-model traces')
    p.add_argument('--receiver-prompt-style', choices=['chat', 'base-completion'], default='chat')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=-1,
                   help='Save every N steps; -1 disables periodic saves (end-of-run checkpoints are still saved)')
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--max-trace-length', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--tasks', nargs='+', default=EVAL)
    p.add_argument('--eval-modes', nargs='+', choices=['unaware', 'aware', 'receiver'], default=['unaware', 'aware', 'receiver'])
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.max_length, args.max_trace_length, args.max_new_tokens) < 1:
        p.error('step counts and token budgets must be positive')
    if args.save_every != -1 and args.save_every < 1:
        p.error('--save-every must be positive or -1 to disable periodic saves')
    torch.manual_seed(args.seed)
    if args.mode == 'preflight':
        preflight(args)
    elif args.mode == 'eval':
        if not args.checkpoint:
            p.error('eval requires --checkpoint')
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
