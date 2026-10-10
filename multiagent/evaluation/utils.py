from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import time
import uuid
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from multiagent.dataset.c2c import apply_chat_template, turn_end_ids
from multiagent.dataset.c2c_types import C2CExample
from multiagent.paths import DATASETS_ROOT
from multiagent.model.unaware_comm import _apply_rope, _base_model

if TYPE_CHECKING:
    from multiagent.model.attention_matching import AttentionMatchingConfig


def eos_ids(tokenizer) -> set[int]:
    return turn_end_ids(tokenizer)


def _pad_token_id(tokenizer) -> int:
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Batched decoding requires a pad_token_id or eos_token_id")
    return int(pad)


def _forward_last_logits(model, **kwargs):
    return model(**kwargs, logits_to_keep=1)


@torch.no_grad()
def _greedy_decode_batch(model, tokenizer, cache, attention_mask, next_ids, next_positions, max_new_tokens, device):
    batch = next_ids.shape[0]
    pad_id = _pad_token_id(tokenizer)
    stop_ids = eos_ids(tokenizer)
    generated = [[] for _ in range(batch)]
    reasons = ["cutoff"] * batch
    active = [True] * batch
    for step in range(max_new_tokens):
        for row, token in enumerate(next_ids.tolist()):
            if not active[row]:
                continue
            if token in stop_ids:
                reasons[row] = "eos"
                active[row] = False
            else:
                generated[row].append(token)
        if not any(active) or step == max_new_tokens - 1:
            break
        tokens = torch.tensor([[t if active[row] else pad_id] for row, t in enumerate(next_ids.tolist())],
                              dtype=torch.long, device=device)
        extension = torch.tensor(active, dtype=attention_mask.dtype, device=device).unsqueeze(1)
        attention_mask = torch.cat([attention_mask, extension], dim=1)
        output = _forward_last_logits(
            model,
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=next_positions.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        next_ids = output.logits[:, -1, :].argmax(dim=-1)
        next_positions = next_positions + 1
    return generated, reasons


DEFAULT_TASKS = ('gsm8k:test', 'math-500:test', 'arc-c:test', 'mmlu-redux:test', 'medqa:test', 'openbookqa:test')


def add_eval_args(parser):
    parser.add_argument("--dataset-root", default=str(DATASETS_ROOT))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Tasks to evaluate in task:split form, e.g. hotpotqa:train.")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Questions decoded together (left-padded greedy decoding). 1 = original serial path. "
                             "Batched BF16 "
                             "decoding is not guaranteed token-identical to serial decoding.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--resume-from",
        help=(
            "Resume from saved evaluation rows without regenerating or regrading them. "
            "Summarization runs also reuse saved summaries and completed prompt variants. "
            "Requires --output."
        ),
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--print-every", type=int, default=0, help="Print one evaluated sample every N absolute samples; 0 disables sample printing.")
    parser.add_argument("--seed", type=int, default=42)


def validate_eval_args(parser, args):
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.resume_from and not args.output:
        parser.error("--resume-from requires --output")


def dtype_from_args(args):
    if args.dtype == "float32":
        return torch.float32
    if args.dtype == "float16":
        return torch.float16
    if args.dtype == "bfloat16":
        return torch.bfloat16
    if args.device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_model(path: Path, dtype: torch.dtype, device: str, adapter: str | Path | None = None):
    from multiagent.model.vanilla_loader import load_causal_model

    model = load_causal_model(path, dtype, device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().to(device).eval()
        print(f"merged adapter: {adapter}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _generate_batch(generate, samples):
    try:
        return generate(samples)
    except torch.cuda.OutOfMemoryError:
        if len(samples) == 1:
            raise
    torch.cuda.empty_cache()
    mid = len(samples) // 2
    print(f"batch of {len(samples)} hit CUDA OOM; retrying as {mid} + {len(samples) - mid}", flush=True)
    return _generate_batch(generate, samples[:mid]) + _generate_batch(generate, samples[mid:])


def add_am_args(parser):
    parser.add_argument("--am-query-samples", type=int, default=512, help="Max reference query vectors per layer/KV-head for Attention Matching.")
    parser.add_argument("--am-compaction-workers", type=int, choices=[1, 2, 4, 8], default=1,
                        help="Experimental independent-head CUDA streams; preserves native AM fits and RNG order.")
    parser.add_argument("--am-nnls-iters", type=int, default=2, help="Projected-gradient NNLS refinement iterations for AM beta fitting.")
    parser.add_argument("--am-score-method", choices=["rms"], default="rms", help="Reference-attention aggregation for compact key selection (only rms; kept for command compatibility).")
    parser.add_argument("--am-head-budget-mode", choices=["uniform", "precomputed"], default="uniform", help="AM per-head budget allocation mode.")
    parser.add_argument("--am-head-budget-path", help="JSON file with precomputed AM head budget shares for head_budget_mode=precomputed.")
    parser.add_argument("--am-chunk-size", type=int, default=0, help="KV-based fixed article chunk size for AM; 0 compacts the full prefix at once.")
    parser.add_argument(
        "--am-min-new-tokens",
        type=int,
        default=0,
        help="Suppress EOS until this many tokens have been generated by Attention Matching.",
    )
    parser.add_argument(
        "--context-cache-max",
        type=int,
        default=0,
        help="Max prepared context caches held on device (0 = unbounded).",
    )


def validate_am_args(parser, args):
    if args.batch_size > 1 and args.am_target_ratio is not None:
        parser.error("--batch-size > 1 is only supported without Attention Matching")
    if args.am_chunk_size < 0:
        parser.error("--am-chunk-size must be nonnegative")
    if not 0 <= args.am_min_new_tokens <= args.max_new_tokens:
        parser.error("--am-min-new-tokens must be between 0 and --max-new-tokens")


def load_head_budget_shares(path: str | None):
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data and all(isinstance(key, str) and key.startswith("L") and "H" in key for key in data):
        entries = {}
        for key, value in data.items():
            layer_text, head_text = key[1:].split("H", 1)
            entries[(int(layer_text), int(head_text))] = float(value)
        num_layers = max(layer for layer, _ in entries) + 1
        num_heads = max(head for _, head in entries) + 1
        return [
            [entries.get((layer, head), 0.0) for head in range(num_heads)]
            for layer in range(num_layers)
        ]
    for key in ("head_budget_shares", "shares", "proportions", "budgets"):
        if key in data:
            return data[key]
    raise ValueError(f"could not find head budget shares in {path}")


def make_am_config(args, **options):
    from multiagent.model.attention_matching import AttentionMatchingConfig

    shares = load_head_budget_shares(args.am_head_budget_path)
    if args.am_head_budget_mode == "precomputed" and shares is None:
        raise ValueError("--am-head-budget-mode precomputed requires --am-head-budget-path")
    if shares is not None and args.am_head_budget_mode != "precomputed":
        raise ValueError("--am-head-budget-path requires --am-head-budget-mode precomputed")
    return AttentionMatchingConfig(
        target_ratio=args.am_target_ratio, query_samples=args.am_query_samples,
        compaction_workers=args.am_compaction_workers, nnls_iters=args.am_nnls_iters,
        score_method=args.am_score_method, head_budget_mode=args.am_head_budget_mode,
        head_budget_shares=shares, chunk_size=args.am_chunk_size,
        generation_reserve=args.max_new_tokens, **options,
    )


def am_metadata(args, config, *, beta_mapping=None, query_transfer=None,
                beyond_context=None, chunk_tokens=None, pack=None, native_window=None):
    return {
        "target_ratio": config.target_ratio,
        "query_samples": config.query_samples,
        "query_sampling_seed": args.seed,
        "query_sampling_seed_scope": "stable_context_sha256",
        "query_style": "paper",
        "nnls_iters": config.nnls_iters,
        "beta_min": config.beta_min,
        "beta_max": config.beta_max,
        "score_method": config.score_method,
        "min_keys": config.min_keys,
        "head_budget_mode": config.head_budget_mode,
        "head_budget_path": args.am_head_budget_path,
        "chunk_size": config.chunk_size,
        "num_chunks": 0,
        "pack": config.pack,
        "packing_window": config.packing_window,
        "generation_reserve": config.generation_reserve,
        "reuse_context_cache_by_context_id": True,
        "min_new_tokens": args.am_min_new_tokens,
        "beta_mapping": beta_mapping,
        "reference_queries": "sender_transferred_off_policy" if query_transfer is not None else "compression_model_on_policy",
        "query_transfer": str(query_transfer) if query_transfer is not None else None,
        "beyond_context": beyond_context,
        "beyond_context_chunk_tokens": chunk_tokens,
        "beyond_context_pack": pack,
        "beyond_context_native_window": native_window,
    }


def interleaved_items(datasets: dict, chunk_size: int, limit_per_dataset: int | None, seed: int):
    indices = {}
    for name, dataset in datasets.items():
        dataset_indices = list(range(len(dataset)))
        if not getattr(dataset, "preserve_order", False):
            shuffle_key = name.replace("+terse", "")
            random.Random(f"{seed}:{shuffle_key}").shuffle(dataset_indices)
        if limit_per_dataset is not None:
            dataset_indices = dataset_indices[:limit_per_dataset]
        indices[name] = dataset_indices

    positions = {name: 0 for name in datasets}
    while True:
        made_progress = False
        for name, dataset in datasets.items():
            end = min(positions[name] + chunk_size, len(indices[name]))
            while positions[name] < end:
                index = indices[name][positions[name]]
                positions[name] += 1
                made_progress = True
                yield name, index, dataset[index]
        if not made_progress:
            break


def summarization_context_id(sample: C2CExample) -> str:
    context_id = str(sample.metadata.get("context_id", "")).strip()
    if context_id:
        return context_id
    context = str(sample.metadata.get("context", ""))
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    return f"{sample.dataset}/{sample.split}/{digest}"


def deterministic_am_context_seed(
    base_seed: int,
    task_key: str,
    index: int,
    sample: C2CExample,
) -> int:
    context_id = str(sample.metadata.get("context_id", "")).strip()
    identity = context_id or f"{task_key}|{int(index)}|{sample.id}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (int(base_seed) + offset) % (2**63 - 1)


def seed_attention_matching_context(
    base_seed: int,
    task_key: str,
    index: int,
    sample: C2CExample,
    device: str,
    *,
    seed_fn=deterministic_am_context_seed,
) -> int:
    seed = seed_fn(base_seed, task_key, index, sample)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    return seed


class BoundedContextCache(dict):

    def __init__(self, capacity: int = 0):
        super().__init__()
        self.capacity = max(0, int(capacity or 0))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.capacity:
            while len(self) > self.capacity:
                oldest = next(iter(self))
                if oldest == key:
                    break
                super().pop(oldest, None)


def synchronized_time(device: str) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class FirstForwardTimer:

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.end = None

    def __enter__(self):
        def finished(module, inputs, output):
            self.end = synchronized_time(self.device)
            self.handle.remove()

        self.handle = self.model.register_forward_hook(finished)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.handle.remove()
        if exc_type is None and self.end is None:
            raise RuntimeError("generation did not execute the timed model forward")


class CompressionJournal:

    def __init__(self, output: str | None, metadata: dict):
        self.path = Path(str(output) + ".measurements.jsonl") if output else None
        self.run_id = uuid.uuid4().hex
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write("run", metadata=metadata)

    def write(self, kind: str, **fields) -> None:
        if self.path is None:
            return
        record = {"run_id": self.run_id, "kind": kind, **fields}
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def renderer_am_token_counts(stop_reason: str, config: dict) -> dict | None:
    if ":renderer+am" not in stop_reason or "/fullqwen" in stop_reason:
        return None
    if config.get("token_count_layout", "full_attention") != "full_attention":
        return None
    span = re.search(r"/span(\d+)-(\d+)of(\d+)", stop_reason)
    if span is None:
        return None
    counts = re.search(r"am(\d+)active/(\d+)allocated/", stop_reason)
    if counts is None:
        raise ValueError("Missing renderer AM cache counts")
    start, end, full = map(int, span.groups())
    heads = int(config["receiver_layers"]) * int(config["receiver_kv_heads"])
    if heads <= 0 or not 0 <= start < end <= full:
        raise ValueError("Invalid renderer dimensions or article span")
    active_entries, allocated_entries = map(int, counts.groups())
    protected = full - (end - start)
    total = active_entries / heads
    article = total - protected
    if not 0 <= article <= end - start or allocated_entries < active_entries:
        raise ValueError("Invalid active/article/allocated cache counts")
    return {
        "schema_version": 1,
        "token_scope": "article",
        "tokens": article,
        "article_positions": article,
        "protected_positions": protected,
        "total_prefix_positions": total,
        "allocated_positions": allocated_entries / heads,
        "original_article_positions": end - start,
        "article_retained_fraction": article / (end - start),
    }


def add_renderer_args(parser):
    parser.add_argument("--renderer", required=True, help="Renderer checkpoint or directory; supplies sender and receiver models")
    parser.add_argument(
        "--renderer-decode-prefix",
        choices=("question", "generation"),
        default="question",
        help="--renderer only. 'question' (default, historical behaviour): the receiver re-reads its own "
        "templated receiver_messages after the rendered cache. 'generation': the receiver gets only the "
        "assistant generation prompt, i.e. it sees the context solely through the rendered cache. "
        "'generation' matches how train_cache_renderer.py trains same_prompt_continuation samples.",
    )
    parser.add_argument("--sample-ids", help="JSON list of sample ids; every other scheduled sample is skipped.")
    parser.add_argument("--sender-adapter", help="PEFT/LoRA adapter directory merged into the renderer's sender model at load time (the renderer itself is unchanged).")


WINDOW_RESERVE = 700


def _resolve_pack_mode(
    pack: str,
    am_config: AttentionMatchingConfig,
    receiver_model,
    article_tokens: int,
    packing_window: int | None = None,
    position_reserve: int = WINDOW_RESERVE,
) -> tuple[AttentionMatchingConfig, float | None]:
    if pack == "clamp":
        return window_capped_budget_config(
            am_config,
            receiver_model,
            article_tokens,
            packing_window=packing_window,
        ), None
    if pack == "fractional":
        window = (
            int(packing_window)
            if packing_window is not None
            else int(_base_model(receiver_model).config.max_position_embeddings)
        )
        if position_reserve >= window:
            raise ValueError(
                f"packing position reserve {position_reserve} must be below window {window}"
            )
        return am_config, float(window - position_reserve)
    raise ValueError(f"unknown beyond-context pack mode: {pack}")


def window_capped_budget_config(
    am_config: AttentionMatchingConfig,
    receiver_model,
    article_tokens: int,
    packing_window: int | None = None,
) -> AttentionMatchingConfig:
    if am_config.head_budget_mode != "precomputed":
        return am_config
    if am_config.head_budget_shares is None:
        raise ValueError("head_budget_mode='precomputed' requires head_budget_shares")
    window = (
        int(packing_window)
        if packing_window is not None
        else int(_base_model(receiver_model).config.max_position_embeddings)
    )
    cap = float(window - WINDOW_RESERVE)
    shares = torch.tensor(am_config.head_budget_shares, dtype=torch.float64)
    total = float(article_tokens) * am_config.target_ratio * shares.numel()
    if total >= cap * shares.numel():
        raise ValueError(
            f"target_ratio {am_config.target_ratio} cannot fit {article_tokens} article "
            f"tokens inside a {window}-position window even with uniform budgets"
        )
    per_head = shares.flatten() / shares.sum() * total
    for _ in range(100):
        over = per_head > cap
        if not over.any():
            break
        excess = float((per_head[over] - cap).sum())
        per_head[over] = cap
        room = per_head < cap
        per_head[room] += excess * per_head[room] / float(per_head[room].sum())
    per_head = per_head.clamp(max=cap)
    clamped = int((torch.tensor(am_config.head_budget_shares, dtype=torch.float64).flatten()
                   / shares.sum() * total > cap).sum())
    if clamped:
        print(
            f"beyond-context budgets: {clamped}/{per_head.numel()} heads clamped at "
            f"{int(cap)} entries over {article_tokens} article tokens; excess water-filled"
        )
    ratios = (per_head / float(article_tokens)).clamp(0.0, 1.0).view(shares.shape)
    return replace(
        am_config,
        head_budget_mode="explicit_ratios",
        head_budget_ratios=ratios.tolist(),
        head_budget_shares=None,
    )


class CompactGridAssembler:

    def __init__(self, base_model, kv_heads: int, span_cap: float | None = None):
        self.base = base_model
        self.num_layers = len(base_model.layers)
        self.kv_heads = kv_heads
        self.span_cap = span_cap
        self._parts: list[list[tuple[Tensor, Tensor, Tensor, Tensor | None]]] = [
            [] for _ in range(self.num_layers)
        ]
        self._max_keep_position = 0.0
        self._repack_started = False

    def add_keep(self, layer_caches: list[tuple[Tensor, Tensor]], rope_positions: Tensor):
        if self._repack_started:
            raise ValueError("keep segments must precede repacked segments")
        if rope_positions.numel() == 0:
            return
        self._max_keep_position = max(self._max_keep_position, float(rope_positions.max().item()))
        for layer_idx in range(self.num_layers):
            key, value = layer_caches[layer_idx]
            if key.shape[2] != rope_positions.numel():
                raise ValueError(
                    f"keep segment length {key.shape[2]} does not match "
                    f"{rope_positions.numel()} rope positions at layer {layer_idx}"
                )
            beta = torch.zeros(
                (self.kv_heads, key.shape[2]), device=key.device, dtype=key.dtype
            )
            self._parts[layer_idx].append((key, value, beta, None))

    def add_repack(
        self,
        layer_caches: list[tuple[Tensor, Tensor]],
        betas: list[Tensor],
        rope_positions: list[Tensor],
    ):
        self._repack_started = True
        for layer_idx in range(self.num_layers):
            key, value = layer_caches[layer_idx]
            self._parts[layer_idx].append(
                (key, value, betas[layer_idx], rope_positions[layer_idx].to(torch.float32))
            )

    @torch.no_grad()
    def assemble(self) -> tuple[DynamicCache, list[Tensor], int]:
        from multiagent.model.attention_matching import _zero_and_validate_inactive_entries

        cache = DynamicCache(config=self.base.config)
        final_betas: list[Tensor] = []
        grid_base = int(self._max_keep_position) + 1 if self._max_keep_position else 0
        decode_start = grid_base
        scaled_heads = 0
        min_scale = 1.0
        for layer_idx in range(self.num_layers):
            layer_dtype = self._parts[layer_idx][0][0].dtype
            totals: Tensor | None = None
            for _key, _value, beta, old_positions in self._parts[layer_idx]:
                if old_positions is not None:
                    active_counts = torch.isfinite(beta).sum(dim=1)
                    totals = active_counts if totals is None else totals + active_counts
            scale = None
            if totals is not None:
                scale = torch.ones(self.kv_heads, dtype=torch.float32, device=totals.device)
                if self.span_cap is not None:
                    span_avail = max(1.0, float(self.span_cap) - grid_base)
                    scale = torch.clamp(span_avail / totals.clamp(min=1).float(), max=1.0)
                    scaled_heads += int((scale < 1.0).sum().item())
                    min_scale = min(min_scale, float(scale.min().item()))

            keys: list[Tensor] = []
            values: list[Tensor] = []
            betas: list[Tensor] = []
            counts: Tensor | None = None
            for key, value, beta, old_positions in self._parts[layer_idx]:
                key = key.to(dtype=layer_dtype)
                value = value.to(dtype=layer_dtype)
                if old_positions is not None:
                    if counts is None:
                        counts = torch.zeros(self.kv_heads, dtype=torch.long, device=key.device)
                    active = torch.isfinite(beta.to(device=key.device))
                    running = torch.cumsum(active.long(), dim=1) - active.long()
                    head_scale = scale.to(device=key.device).unsqueeze(1)
                    new_positions = (
                        grid_base + (counts.unsqueeze(1) + running).float() * head_scale
                    ).unsqueeze(0)
                    deroped = _apply_rope(self.base, key, layer_idx, old_positions, inverse=True)
                    key = _apply_rope(self.base, deroped, layer_idx, new_positions, inverse=False)
                    counts = counts + active.sum(dim=1)
                keys.append(key)
                values.append(value)
                betas.append(beta.to(dtype=key.dtype))
            layer_key = torch.cat(keys, dim=2)
            layer_value = torch.cat(values, dim=2)
            layer_beta = torch.cat(betas, dim=1)
            _zero_and_validate_inactive_entries(layer_key, layer_value, layer_beta)
            cache.update(layer_key, layer_value, layer_idx)
            final_betas.append(layer_beta)
            if counts is not None and bool((counts > 0).any()):
                last_offset = (
                    (counts - 1).clamp(min=0).float() * scale.to(counts.device)
                ).max().item()
                layer_decode_start = grid_base + int(last_offset) + 1
                decode_start = max(decode_start, layer_decode_start)
        if scaled_heads:
            print(
                f"beyond-context fractional packing: {scaled_heads} layer-heads scaled "
                f"(min spacing {min_scale:.3f}) into span cap {self.span_cap}"
            )
        return cache, final_betas, decode_start


def _lookup_rope_positions(lookup: Tensor, index_positions: list[Tensor]) -> list[Tensor]:
    return [lookup[indices.clamp(min=0, max=lookup.numel() - 1)] for indices in index_positions]


_HEADER_LEN_CACHE: dict[int, int] = {}


def _receiver_header_len(tokenizer) -> int:
    key = id(tokenizer)
    if key not in _HEADER_LEN_CACHE:
        sentinel = "XQZC2CSENTINELZQX"
        text = apply_chat_template(
            tokenizer, [{"role": "user", "content": sentinel}], add_generation_prompt=False
        )
        head = text[: text.find(sentinel)]
        _HEADER_LEN_CACHE[key] = len(
            tokenizer(head, add_special_tokens=False)["input_ids"]
        )
    return _HEADER_LEN_CACHE[key]
