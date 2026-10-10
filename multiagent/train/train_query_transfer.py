from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import C2CPhase1Collator
from multiagent.dataset.c2c_cross_family import (
    REPEAT_INSTRUCTION,
    CrossFamilyPhase1Collator,
    _pad_stack,
)
from multiagent.dataset.c2c_registry import load_c2c_dataset_specs
from multiagent.model.checkpoints import c2c_model_names, load_checkpoint_state
from multiagent.model.query_transfer import (
    QueryToQueryTransfer,
    load_query_transfer_state_dict,
    query_transfer_checkpoint_file,
)
from multiagent.model.unaware_comm import _base_model
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.paths import DATASETS_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sender-to-receiver self-study query transfer for receiver-space AM"
    )
    parser.add_argument("--c2c", help="Existing K/V C2C checkpoint or output directory")
    parser.add_argument("--sender-model", help="Sender base model (use with --receiver-model instead of --c2c)")
    parser.add_argument("--receiver-model", help="Receiver base model (use with --sender-model instead of --c2c)")
    parser.add_argument(
        "--architecture",
        choices=("linear", "mlp"),
        required=True,
        help="Per-layer/head query projector architecture.",
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=8,
        help="MLP hidden width multiplier over sender head dimension (default: 8).",
    )
    parser.add_argument(
        "--pairing-file",
        help=(
            "Learned query pairing JSON (train/head_pairings scorer output on query "
            "features): per receiver (layer, head) sender (layer, head) source. "
            "Default: proportional layer/head maps."
        ),
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--dataset-root", default=str(DATASETS_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--query-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="constant",
        help="constant keeps --learning-rate fixed (legacy); cosine warms up linearly then decays to 0.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps (cosine scheduler only).")
    parser.add_argument(
        "--lr-total-steps",
        type=int,
        help="Cosine horizon in optimizer steps (default: total epochs x dataset length, counting resumed epochs).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--resume-from",
        help=(
            "Query-transfer checkpoint file or directory to resume. --epochs specifies "
            "additional epochs to run."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = [item for value in args.datasets for item in value.split(",") if item]
    explicit_models = args.sender_model is not None or args.receiver_model is not None
    if args.c2c and explicit_models:
        parser.error("use either --c2c or --sender-model/--receiver-model, not both")
    if not args.c2c and not (args.sender_model and args.receiver_model):
        parser.error("provide --c2c or both --sender-model and --receiver-model")
    if args.batch_size != 1:
        parser.error("query-transfer training currently requires --batch-size 1")
    if args.query_samples <= 0:
        parser.error("--query-samples must be positive")
    if args.expansion <= 0:
        parser.error("--expansion must be positive")
    return args


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


def load_model(path: Path, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def _messages_key(messages) -> tuple:
    return tuple(
        tuple(sorted((str(key), str(value)) for key, value in message.items()))
        for message in messages
    )


class UniqueSenderContextDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self.original_rows = len(dataset)
        self.rows = []
        seen = set()
        for row in dataset:
            key = _messages_key(row.sender_messages)
            if key in seen:
                continue
            seen.add(key)
            self.rows.append(replace(row, phase1_messages=row.sender_messages))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def model_query_shape(model) -> tuple[int, int, int]:
    base = _base_model(model)
    config = base.config
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // heads))
    return layers, heads, head_dim


@torch.no_grad()
def collect_queries(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    query_batch: torch.Tensor,
    query_pos: torch.Tensor,
) -> list[torch.Tensor]:
    base = _base_model(model)
    captured: list[torch.Tensor | None] = [None] * len(base.layers)
    handles = []

    for layer_idx, layer in enumerate(base.layers):
        attention = layer.self_attn

        def capture(_module, _inputs, output, *, index=layer_idx, attn=attention):
            selected = output[query_batch, query_pos]
            selected = selected.view(selected.shape[0], -1, attn.head_dim)
            q_norm = getattr(attn, "q_norm", None)
            if q_norm is not None:
                selected = q_norm(selected)
            captured[index] = selected.detach().to(device="cpu")

        handles.append(attention.q_proj.register_forward_hook(capture))

    try:
        output = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        del output
    finally:
        for handle in handles:
            handle.remove()

    if any(query is None for query in captured):
        raise RuntimeError("failed to capture queries from every transformer layer")
    return [query for query in captured if query is not None]


def cross_family_query_batch(
    batch: dict,
    sender_instruction_ids: torch.Tensor,
    receiver_instruction_len: int,
    limit: int,
    seed: int,
    sender_pad_id: int,
):
    align_mask = batch["align_mask"]
    pairs = align_mask.nonzero(as_tuple=False)
    if not pairs.numel():
        return None
    if pairs.shape[0] > limit:
        generator = torch.Generator(device=pairs.device)
        generator.manual_seed(seed)
        order = torch.randperm(pairs.shape[0], generator=generator, device=pairs.device)[:limit]
        pairs = pairs[order]
    device = batch["sender"]["input_ids"].device
    instruction = sender_instruction_ids.to(device)

    sender_row_masks = batch["sender"]["attention_mask"].bool()
    sender_contexts = [
        ids[mask] for ids, mask in zip(batch["sender"]["input_ids"], sender_row_masks)
    ]
    rows, masks, position_rows = [], [], []
    for context in sender_contexts:
        repeated = torch.cat([context, instruction, context])
        rows.append(repeated)
        masks.append(torch.ones_like(repeated))
        position_rows.append(torch.arange(repeated.numel(), dtype=torch.long, device=device))
    sender_repeat_inputs = {
        "input_ids": _pad_stack(rows, sender_pad_id, device),
        "attention_mask": _pad_stack(masks, 0, device),
        "position_ids": _pad_stack(position_rows, 0, device),
    }
    receiver_repeat_inputs = {
        key: value for key, value in batch["receiver_repeat"].items() if key != "query_mask"
    }

    query_batch = pairs[:, 0]
    slot = pairs[:, 1]
    sender_ctx_lens = batch["sender"]["attention_mask"].sum(dim=1).long().to(device)
    receiver_ctx_lens = batch["receiver"]["attention_mask"].sum(dim=1).long().to(device)
    sender_query_pos = (
        sender_ctx_lens[query_batch]
        + instruction.numel()
        + batch["align_sender"].to(device)[query_batch, slot]
    )
    receiver_query_pos = (
        receiver_ctx_lens[query_batch]
        + receiver_instruction_len
        + batch["align_receiver"].to(device)[query_batch, slot]
    )
    return (
        sender_repeat_inputs,
        receiver_repeat_inputs,
        query_batch,
        sender_query_pos,
        receiver_query_pos,
    )


def sampled_query_positions(query_mask: torch.Tensor, limit: int, seed: int):
    indices = query_mask.nonzero(as_tuple=False)
    if not indices.numel():
        raise ValueError("self-study repeat sequence has no query positions")
    if indices.shape[0] > limit:
        generator = torch.Generator(device=indices.device)
        generator.manual_seed(seed)
        order = torch.randperm(indices.shape[0], generator=generator, device=indices.device)[:limit]
        indices = indices[order]
    return indices[:, 0], indices[:, 1]


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as stream:
        json.dump(value, stream, indent=2)
    tmp.replace(path)


def save_checkpoint(
    output_dir: Path,
    name: str,
    query_transfer,
    optimizer,
    step: int,
    epoch: int,
    args,
    losses,
    c2c_checkpoint: Path | None,
    sender_model: Path | None = None,
    receiver_model: Path | None = None,
):
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "query_transfer_state_dict": query_transfer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "torch_rng_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
            "num_layers": query_transfer.num_layers,
            "num_heads": query_transfer.num_heads,
            "head_dim": query_transfer.head_dim,
            "sender_layers": query_transfer.sender_layers,
            "sender_heads": query_transfer.sender_heads,
            "sender_head_dim": query_transfer.sender_head_dim,
            "receiver_layers": query_transfer.receiver_layers,
            "receiver_heads": query_transfer.receiver_heads,
            "receiver_head_dim": query_transfer.receiver_head_dim,
            "architecture": query_transfer.architecture,
            "expansion": query_transfer.expansion,
            "pairing": query_transfer.pairing,
            "c2c_checkpoint": None if c2c_checkpoint is None else str(c2c_checkpoint),
            "sender_model": None if sender_model is None else str(sender_model),
            "receiver_model": None if receiver_model is None else str(receiver_model),
            "args": vars(args),
        },
        checkpoint_dir / "query_transfer.pt",
    )
    save_json(output_dir / "losses.json", losses)
    save_json(
        output_dir / "latest.json",
        {"checkpoint": name, "step": step, "epoch": epoch, "time": time.time()},
    )


def _load_loss_history(checkpoint_path: Path, output_dir: Path, state: dict) -> list[dict]:
    if isinstance(state.get("losses"), list):
        return state["losses"]
    candidates = [
        output_dir / "losses.json",
        checkpoint_path.parent.parent / "losses.json",
        checkpoint_path.parent / "losses.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as stream:
                value = json.load(stream)
            if isinstance(value, list):
                return value
    return []


def load_training_checkpoint(path, query_transfer, optimizer, device: str, output_dir: Path):
    checkpoint_path = query_transfer_checkpoint_file(path)
    state = torch.load(checkpoint_path, map_location=device)
    load_query_transfer_state_dict(query_transfer, state["query_transfer_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    return (
        checkpoint_path,
        int(state.get("step", 0)),
        int(state.get("epoch", 0)),
        _load_loss_history(checkpoint_path, output_dir, state),
    )


def main():
    args = parse_args()
    args.query_training_messages = "sender_messages"
    args.deduplicate_sender_messages = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "args.json", vars(args))

    dtype = dtype_from_args(args)
    c2c_path = None
    if args.c2c:
        c2c_path, c2c_state = load_checkpoint_state(args.c2c)
        sender_name, receiver_name = c2c_model_names(c2c_state, c2c_path)
        print(f"c2c_checkpoint: {c2c_path}")
    else:
        sender_name, receiver_name = args.sender_model, args.receiver_model
        print("c2c_checkpoint: none (explicit base models)")
    sender_path = resolve_model_path(sender_name)
    receiver_path = resolve_model_path(receiver_name)
    print(f"sender_model: {sender_path}")
    print(f"receiver_model: {receiver_path}")
    print(f"device={args.device} dtype={dtype}")

    sender_tokenizer = load_tokenizer(sender_path)
    receiver_tokenizer = load_tokenizer(receiver_path)
    probe_text = "The 3 quick brown foxes jumped over 2 lazy dogs." + REPEAT_INSTRUCTION
    same_tokenization = (
        len(sender_tokenizer) == len(receiver_tokenizer)
        and sender_tokenizer(probe_text, add_special_tokens=False)["input_ids"]
        == receiver_tokenizer(probe_text, add_special_tokens=False)["input_ids"]
    )
    print(f"tokenization: {'shared (self-study path)' if same_tokenization else 'differs (char-aligned path)'}")
    sender_model = load_model(sender_path, dtype, args.device)
    receiver_model = load_model(receiver_path, dtype, args.device)
    sender_shape = model_query_shape(sender_model)
    receiver_shape = model_query_shape(receiver_model)
    pairing = None
    if args.pairing_file:
        with open(args.pairing_file) as stream:
            pairing_data = json.load(stream)
        pairing = (
            pairing_data.get("query_source") or pairing_data.get("key_source")
            if isinstance(pairing_data, dict)
            else pairing_data
        )
        if pairing is None:
            raise ValueError(f"no query_source/key_source assignment in {args.pairing_file}")
        print(f"pairing_file: {args.pairing_file}")
    query_transfer = QueryToQueryTransfer(
        *sender_shape,
        *receiver_shape,
        architecture=args.architecture,
        expansion=args.expansion,
        pairing=pairing,
    ).to(
        args.device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        query_transfer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    source_dataset = load_c2c_dataset_specs(
        args.datasets,
        root=args.dataset_root,
        start=args.start,
        limit_per_dataset=args.limit_per_dataset,
    )
    dataset = UniqueSenderContextDataset(source_dataset)
    if same_tokenization:
        collator = C2CPhase1Collator(
            sender_tokenizer=sender_tokenizer,
            receiver_tokenizer=receiver_tokenizer,
            max_length=args.max_length,
            device=args.device,
            receiver_device=args.device,
            include_repeat_prefill=True,
        )
    else:
        collator = CrossFamilyPhase1Collator(
            sender_tokenizer=sender_tokenizer,
            receiver_tokenizer=receiver_tokenizer,
            max_length=args.max_length,
            device=args.device,
            receiver_device=args.device,
            include_repeat=True,
        )
        sender_repeat_instruction = torch.tensor(
            sender_tokenizer(REPEAT_INSTRUCTION, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=False,
    )
    print(
        f"dataset rows: {dataset.original_rows} QA rows -> "
        f"{len(dataset)} unique sender contexts "
        f"({dataset.original_rows - len(dataset)} duplicates removed)"
    )
    print(f"trainable params: {sum(p.numel() for p in query_transfer.parameters()):,}")
    print(f"output_dir: {output_dir}")

    losses = []
    rolling = deque(maxlen=100)
    step = 0
    current_epoch = 0
    starting_epoch = 0
    if args.resume_from:
        resume_path, step, starting_epoch, losses = load_training_checkpoint(
            args.resume_from,
            query_transfer,
            optimizer,
            args.device,
            output_dir,
        )
        current_epoch = starting_epoch
        for row in losses[-100:]:
            rolling.append(float(row["loss"]))
        print(
            f"resumed query transfer from {resume_path} at "
            f"step={step}, completed_epochs={starting_epoch}, losses={len(losses)}"
        )

    final_epoch = starting_epoch + args.epochs
    schedule_total = args.lr_total_steps or final_epoch * len(loader)
    if args.lr_scheduler == "cosine" and args.warmup_steps >= schedule_total:
        raise ValueError(
            f"--warmup-steps {args.warmup_steps} must be below the LR horizon {schedule_total}"
        )

    def learning_rate_at(step_index: int) -> float:
        if args.lr_scheduler == "constant":
            return args.learning_rate
        if args.warmup_steps > 0 and step_index <= args.warmup_steps:
            return args.learning_rate * step_index / args.warmup_steps
        span = max(1, schedule_total - args.warmup_steps)
        progress = min(1.0, (step_index - args.warmup_steps) / span)
        return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    try:
        query_transfer.train()
        for epoch_offset in range(args.epochs):
            current_epoch = starting_epoch + epoch_offset + 1
            progress = tqdm(loader, desc=f"query-transfer {current_epoch}/{final_epoch}")
            for batch in progress:
                if batch is None:
                    continue
                if same_tokenization:
                    sender_inputs = batch["sender"]
                    receiver_inputs = batch["receiver"]
                    if not torch.equal(sender_inputs["input_ids"], receiver_inputs["input_ids"]):
                        raise ValueError("sender and receiver context tokenization differs")
                    repeat = batch["receiver_repeat"]
                    query_batch, query_pos = sampled_query_positions(
                        repeat["query_mask"],
                        args.query_samples,
                        args.seed + step,
                    )
                    repeat_inputs = {
                        key: value
                        for key, value in repeat.items()
                        if key != "query_mask"
                    }
                    sender_queries = collect_queries(
                        sender_model,
                        query_batch=query_batch,
                        query_pos=query_pos,
                        **repeat_inputs,
                    )
                    receiver_queries = collect_queries(
                        receiver_model,
                        query_batch=query_batch,
                        query_pos=query_pos,
                        **repeat_inputs,
                    )
                    source_token_count = int(sender_inputs["attention_mask"].sum().item())
                    repeat_token_count = int(repeat["attention_mask"].sum().item())
                else:
                    prepared = cross_family_query_batch(
                        batch,
                        sender_repeat_instruction,
                        len(collator.repeat_instruction_ids),
                        args.query_samples,
                        args.seed + step,
                        sender_tokenizer.pad_token_id,
                    )
                    if prepared is None:
                        continue
                    (
                        sender_repeat_inputs,
                        receiver_repeat_inputs,
                        query_batch,
                        sender_query_pos,
                        receiver_query_pos,
                    ) = prepared
                    sender_queries = collect_queries(
                        sender_model,
                        query_batch=query_batch,
                        query_pos=sender_query_pos,
                        **sender_repeat_inputs,
                    )
                    receiver_queries = collect_queries(
                        receiver_model,
                        query_batch=query_batch,
                        query_pos=receiver_query_pos,
                        **receiver_repeat_inputs,
                    )
                    source_token_count = int(batch["sender"]["attention_mask"].sum().item())
                    repeat_token_count = int(batch["receiver_repeat"]["attention_mask"].sum().item())

                optimizer.zero_grad(set_to_none=True)
                sender_queries = [
                    query.to(args.device, dtype=torch.float32) for query in sender_queries
                ]
                layer_losses = []
                for layer_idx, receiver_q in enumerate(receiver_queries):
                    receiver_q = receiver_q.to(args.device, dtype=torch.float32)
                    layer_loss = F.mse_loss(
                        query_transfer.forward_layer_all(layer_idx, sender_queries),
                        receiver_q,
                    )
                    (layer_loss / query_transfer.num_layers).backward()
                    layer_losses.append(float(layer_loss.detach().cpu()))
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    query_transfer.parameters(),
                    args.max_grad_norm,
                )
                lr_value = learning_rate_at(step + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr_value
                optimizer.step()

                step += 1
                loss_value = sum(layer_losses) / len(layer_losses)
                rolling.append(loss_value)
                row = {
                    "step": step,
                    "epoch": current_epoch,
                    "loss": loss_value,
                    "loss_avg_100": sum(rolling) / len(rolling),
                    "min_layer_loss": min(layer_losses),
                    "max_layer_loss": max(layer_losses),
                    "query_samples": int(query_batch.numel()),
                    "source_tokens": source_token_count,
                    "repeat_tokens": repeat_token_count,
                    "learning_rate": lr_value,
                    "grad_norm": float(grad_norm),
                    "time": time.time(),
                }
                losses.append(row)
                progress.set_postfix(loss=f"{row['loss_avg_100']:.6f}")
                if args.report_every > 0 and step % args.report_every == 0:
                    print(json.dumps(row), flush=True)
                    save_json(output_dir / "losses.json", losses)
                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        output_dir,
                        f"checkpoint-{step}",
                        query_transfer,
                        optimizer,
                        step,
                        current_epoch,
                        args,
                        losses,
                        c2c_path,
                        sender_path,
                        receiver_path,
                    )
                if args.max_steps is not None and step >= args.max_steps:
                    raise StopIteration
                del sender_queries, receiver_queries
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        save_checkpoint(
            output_dir,
            f"exit-checkpoint-{step}",
            query_transfer,
            optimizer,
            step,
            max(1, current_epoch),
            args,
            losses,
            c2c_path,
            sender_path,
            receiver_path,
        )
        print(f"saved query transfer at step {step}", flush=True)


if __name__ == "__main__":
    main()
