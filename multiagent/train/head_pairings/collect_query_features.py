#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c_cross_family import CrossFamilyPhase1Collator
from multiagent.dataset.c2c_registry import load_c2c_dataset_specs
from multiagent.model.unaware_comm import _base_model
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.train.train_utils import load_model, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sender-model", default="Qwen3-4B")
    parser.add_argument("--receiver-model", default="Qwen3-14B")
    parser.add_argument("--datasets", nargs="+", default=["qasper+nox:train", "quality:train"])
    parser.add_argument("--limit-per-dataset", type=int, default=4000)
    parser.add_argument("--num-contexts", type=int, default=128, help="Contexts (documents) to collect from.")
    parser.add_argument("--pairs-per-context", type=int, default=128, help="Aligned token pairs sampled per context.")
    parser.add_argument("--min-aligned", type=int, default=64, help="Skip contexts with fewer aligned pairs.")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--receiver-device", default="cuda:1")
    parser.add_argument("--output", default="outputs/head_pairings_queries/features.pt")
    return parser.parse_args()


def query_shape(model) -> tuple[int, int, int]:
    base = _base_model(model)
    config = base.config
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // heads))
    return layers, heads, head_dim


@torch.no_grad()
def query_features(model, inputs: dict, token_index: torch.Tensor) -> torch.Tensor:
    base = _base_model(model)
    captured: list[torch.Tensor | None] = [None] * len(base.layers)
    handles = []
    for layer_idx, layer in enumerate(base.layers):
        attention = layer.self_attn

        def capture(_module, _inputs, output, *, index=layer_idx, attn=attention):
            queries = output.view(output.shape[0], output.shape[1], -1, attn.head_dim)
            q_norm = getattr(attn, "q_norm", None)
            if q_norm is not None:
                queries = q_norm(queries)
            selected = queries[0, token_index.to(queries.device)]
            captured[index] = selected.to(torch.float16).cpu()

        handles.append(attention.q_proj.register_forward_hook(capture))
    try:
        forward_inputs = {**inputs, "use_cache": False}
        try:
            model(**forward_inputs, logits_to_keep=1)
        except TypeError:
            model(**forward_inputs)
    finally:
        for handle in handles:
            handle.remove()
    if any(query is None for query in captured):
        raise RuntimeError("failed to capture queries from every transformer layer")
    return torch.stack(captured, dim=1)


def main():
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    sender_path = resolve_model_path(args.sender_model)
    receiver_path = resolve_model_path(args.receiver_model)
    sender_tokenizer = load_tokenizer(sender_path)
    receiver_tokenizer = load_tokenizer(receiver_path)
    sender_model = load_model(sender_path, dtype, args.device)
    receiver_model = load_model(receiver_path, dtype, args.receiver_device)
    sender_shape = query_shape(sender_model)
    receiver_shape = query_shape(receiver_model)
    print(f"sender {args.sender_model}: layers x q-heads x dim = {sender_shape}")
    print(f"receiver {args.receiver_model}: layers x q-heads x dim = {receiver_shape}")

    dataset = load_c2c_dataset_specs(args.datasets, include_answer_message=False, limit_per_dataset=args.limit_per_dataset)
    collator = CrossFamilyPhase1Collator(
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        max_length=args.max_length,
        device=args.device,
        receiver_device=args.receiver_device,
        position_mode="interp",
        include_repeat=False,
    )
    order = torch.randperm(len(dataset), generator=generator).tolist()

    sender_queries, receiver_queries, doc_rows = [], [], []
    contexts = 0
    skipped = 0
    aligned_total = 0
    started = time.time()
    progress = tqdm(total=args.num_contexts, desc="contexts")
    for dataset_index in order:
        if contexts >= args.num_contexts:
            break
        batch = collator([dataset[dataset_index]])
        if batch is None or batch["aligned_tokens"] < args.min_aligned:
            skipped += 1
            continue

        align_sender = batch["align_sender"][0].cpu()
        align_receiver = batch["align_receiver"][0].cpu()
        n_aligned = int(batch["align_mask"][0].sum())
        pick = torch.randperm(n_aligned, generator=generator)[: args.pairs_per_context]
        sender_index = align_sender[pick]
        receiver_index = align_receiver[pick]

        sender_queries.append(query_features(sender_model, batch["sender"], sender_index))
        receiver_queries.append(query_features(receiver_model, batch["receiver"], receiver_index))
        doc_rows.append(torch.full((len(pick),), contexts, dtype=torch.int32))

        aligned_total += len(pick)
        contexts += 1
        progress.update(1)
        progress.set_postfix(pairs=aligned_total, skipped=skipped)
    progress.close()
    if contexts == 0:
        raise RuntimeError("no usable contexts collected")

    sender_q = torch.cat(sender_queries)
    receiver_q = torch.cat(receiver_queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sender_key": sender_q,
            "sender_value": sender_q,
            "receiver_key": receiver_q,
            "receiver_value": receiver_q,
            "doc_index": torch.cat(doc_rows),
            "meta": {
                "sender_model": args.sender_model,
                "receiver_model": args.receiver_model,
                "sender_shape": tuple(sender_shape),
                "receiver_shape": tuple(receiver_shape),
                "contexts": contexts,
                "pairs": aligned_total,
                "skipped": skipped,
                "kind": "query",
                "args": vars(args),
            },
        },
        output_path,
    )
    print(
        f"wrote {output_path}: {aligned_total} aligned pairs from {contexts} contexts "
        f"({skipped} skipped) in {time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
