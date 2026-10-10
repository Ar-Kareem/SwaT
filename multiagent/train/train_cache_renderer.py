from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import chat_prompt_ids, generation_prompt_ids
from multiagent.dataset.c2c_cross_family import (
    CrossFamilyNextTokenCollator,
    CrossFamilyPhase1Collator,
    encode_context_pair,
    _pad_stack,
)
from multiagent.dataset.c2c_registry import load_c2c_dataset_specs, load_c2c_task
from multiagent.dataset.c2c_token_alignment import alignment_config, checkpoint_alignment
from multiagent.model.cache_renderer import (
    CacheRendererHeadwise,
    CacheRendererPointwise,
    CacheRenderer,
    CacheRendererV2,
    flatten_sender_cache,
    pred_to_cache,
    save_renderer_checkpoint,
)
from multiagent.model.dense_prefill import dense_prefill_logits
from multiagent.model.unaware_comm import _apply_rope, _base_model, cache_layer, full_sender_cache, kv_layer_indices, tuple_to_dynamic_cache
from multiagent.train.lr_scheduler import add_lr_scheduler_args, build_lr_scheduler, resolve_total_steps
from multiagent.train.train_utils import (
    cycler,
    eos_ids,
    load_model,
    load_tokenizer,
    metric_from_grade,
    move_kv_cache,
    prefill_all,
    save_json,
    save_losses_and_plot,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cross-family cache renderer (prefill emulator)")
    parser.add_argument("--sender-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--exclude-answer-message", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-trace-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer update. Each micro-batch draws its own type (kv/ce/dense) by the usual "
        "fractions, so one update averages CE and dense gradients instead of alternating them. --max-steps, "
        "--save-every and --downstream-eval-every count optimizer updates.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    add_lr_scheduler_args(parser)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--grad-skip-threshold",
        type=float,
        default=0.0,
        help="Skip the optimizer step entirely when the (pre-clip) grad norm exceeds this; 0 disables. "
        "CE batches through an 8k-token rendered cache occasionally produce 100-1000x grad-norm spikes "
        "even at tiny loss; clipping still lets a burst of such steps damage the weights.",
    )
    parser.add_argument("--arch", choices=("v1", "v2", "pointwise", "headwise", "dense-latent"), default="v2", help="v1 = flat trunk + single output head; v2 = depth-aligned blocks (per-block receiver-layer heads, per-block sender-layer-slice cross-attention); pointwise = ABLATION, v2 without any token mixing (each receiver token rendered from its embedding + the one aligned sender token; widen --ffn-dim to 8192 to match v2's parameter count); headwise = ABLATION, C2C-style one MLP per receiver (layer, head) from the depth-interpolated matched sender head (see --headwise-hidden).")
    parser.add_argument("--headwise-hidden", type=int, default=2048, help="headwise arch: hidden width of each per-head MLP (2048 -> ~269M params total for 32x8 heads).")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=8, help="Trunk blocks (v2: uneven receiver_layers are split proportionally).")
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument(
        "--bound-cache-stats",
        action="store_true",
        help="v2 only: hard-project every predicted (layer, head) K/V RMS to calibrated receiver "
        "statistics inside the module — keeps the rendered cache on the plausible-statistics "
        "manifold so CE-through-cache gradients stay well-conditioned.",
    )
    parser.add_argument("--stat-calibration-batches", type=int, default=8, help="Receiver-prefill batches used to fill the bound-cache-stats RMS buffers.")
    parser.add_argument(
        "--bound-stats-exact-grad",
        action="store_true",
        help="Backprop through the bounding RMS denominator (true Jacobian, projects out the "
        "radial component) instead of detaching it. The detached backward is the gradient of "
        "a different function than the forward — suspected cause of CE/KL loss RISING under "
        "its own gradient descent.",
    )
    parser.add_argument(
        "--no-sender-memory",
        action="store_true",
        help="Ablation: render from receiver embeddings only (pure prefill emulator, no cross-attention).",
    )
    parser.add_argument(
        "--causal-cross-attention",
        action="store_true",
        help="Causal renderer cross-attention: receiver query i attends sender token j only if j's "
        "content ends at or before receiver token i's end (staircase from the char map). Applied in "
        "every batch type and at deployment. Prerequisite for --dense-batch-fraction; makes "
        "render(CTX)[:j] == render(CTX[:j]) exactly, so deployed caches are incrementally extendable.",
    )
    parser.add_argument(
        "--dense-batch-fraction",
        type=float,
        default=0.0,
        help="Probability that a CE-phase draw is a dense-prefill batch (every-position CE through the "
        "rendered cache) instead of an answer-CE batch. 0 disables; 1 replaces answer CE entirely.",
    )
    parser.add_argument(
        "--dense-datasets",
        nargs="+",
        help="Dataset specs for dense batches (contexts only; answers unused). Defaults to --datasets. "
        "Unlabelled corpora work here — dense supervision needs no QA pairs.",
    )
    parser.add_argument(
        "--dense-max-length",
        type=int,
        default=4096,
        help="Context length cap for dense batches. The dense forward backprops through all receiver "
        "layers at full document length, so its memory ceiling is well above the answer-CE batches'.",
    )
    parser.add_argument(
        "--dense-w-choices",
        nargs="+",
        type=int,
        default=[0],
        help="Real-token band widths sampled per dense batch: position i uses its own true KV for the "
        "last w+1 positions and the rendered cache before them. w=0 is the pure one-real-token scheme; "
        "sampling larger w also trains the multi-real-token decode regime.",
    )
    parser.add_argument(
        "--dense-loss",
        choices=("hard",),
        default="hard",
        help="Dense-batch objective: CE against the document's own next tokens (only hard; kept for command compatibility).",
    )
    parser.add_argument("--phase-schedule", choices=("kv-only", "mixed"), default="mixed")
    parser.add_argument("--ce-batch-fraction", type=float, default=0.5, help="CE draw probability (mixed schedule only).")
    parser.add_argument(
        "--ce-loss",
        choices=("hard",),
        default="hard",
        help="CE-batch objective: teacher-forced cross-entropy on trace labels (only hard; kept for command compatibility).",
    )
    parser.add_argument("--position-mode", default="interp")
    parser.add_argument("--token-alignment", choices=("native", "c2c-longest"), default="native")
    parser.add_argument("--downstream-eval-every", type=int, default=0)
    parser.add_argument("--downstream-eval-datasets", nargs="+")
    parser.add_argument("--downstream-eval-samples", type=int, default=64)
    parser.add_argument("--downstream-eval-strides", nargs="+", type=int)
    parser.add_argument("--downstream-generate-samples", type=int, default=64)
    parser.add_argument("--downstream-max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--receiver-device", help="Device for receiver model and renderer; defaults to --device")
    parser.add_argument("--renderer-device", help="Device for the renderer and its optimizer state; defaults to "
                        "--receiver-device. Point it at the sender GPU when the receiver alone nearly fills its "
                        "card (e.g. 12B+ receivers) — activations shuttle across, weights/grads/Adam stay off.")
    parser.add_argument("--report-every", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=-1,
                        help="Save every N optimizer steps; -1 disables periodic saves (exit checkpoint is still saved).")
    parser.add_argument("--loss-window", type=int, default=100)
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--init-weights-from",
        help="Load ONLY renderer weights from this checkpoint dir (fresh optimizer/scheduler/step) — "
        "for restarting a phase with different hyperparameters.",
    )
    parser.add_argument("--oom-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-max-receiver-tokens", type=int, default=None,
                        help="OOM preflight only: replace the synthetic split-prompt CE shapes with splits of this many "
                        "receiver tokens (context + decode). Set from a measured pool maximum; not enforced in training.")
    parser.add_argument("--checkpoint-receiver", action="store_true",
                        help="Recompute frozen receiver decoder activations during CE backward to reduce memory.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run setup and the worst-case OOM preflight, then exit before training.",
    )
    parser.add_argument("--empty-cache-every", type=int, default=0,
                        help="torch.cuda.empty_cache() every N optimizer steps (0 = never). "
                        "Combats allocator fragmentation on tight-memory receivers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.save_every != -1 and args.save_every < 1:
        parser.error("--save-every must be positive or -1 to disable periodic saves")
    args.datasets = [item for value in args.datasets for item in value.split(",") if item]
    args.receiver_device = args.receiver_device or args.device
    args.renderer_device = args.renderer_device or args.receiver_device
    if args.renderer_device != args.receiver_device:
        torch.autograd.set_multithreading_enabled(False)
        print("renderer on", args.renderer_device, "-> single-threaded autograd backward")
    if args.downstream_eval_every > 0 and not args.downstream_eval_datasets:
        parser.error("--downstream-eval-every requires --downstream-eval-datasets")
    if not 0.0 <= args.dense_batch_fraction <= 1.0:
        parser.error("--dense-batch-fraction must be in [0, 1]")
    if args.dense_batch_fraction > 0 and not args.causal_cross_attention:
        parser.error("--dense-batch-fraction requires --causal-cross-attention (a non-causal renderer makes the dense objective trivially satisfiable)")
    if args.dense_batch_fraction > 0 and args.no_sender_memory:
        parser.error("--dense-batch-fraction is incompatible with --no-sender-memory")
    if any(w < 0 for w in args.dense_w_choices):
        parser.error("--dense-w-choices must be non-negative")
    if args.token_alignment == "c2c-longest":
        if args.arch != "v2" or not args.causal_cross_attention or args.no_sender_memory:
            parser.error("--token-alignment c2c-longest requires v2, sender memory and causal cross-attention")
        if args.position_mode != "interp":
            parser.error("--position-mode only applies to native alignment")
        if args.batch_size != 1:
            parser.error("C2C-longest currently supports batch size 1")
    return args


def _autocast(device: str):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))


class _SyncedTransfer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, destination):
        ctx.source = tensor.device
        torch.cuda.synchronize(tensor.device)
        moved = tensor.to(destination)
        torch.cuda.synchronize(destination)
        return moved

    @staticmethod
    def backward(ctx, grad):
        torch.cuda.synchronize(grad.device)
        moved = grad.to(ctx.source)
        torch.cuda.synchronize(ctx.source)
        return moved, None


def run_renderer(args, renderer, embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask=None, output_device=None):
    device = args.renderer_device
    def move(t):
        return t.to(device) if torch.is_tensor(t) else t
    with _autocast(device):
        pred = renderer(move(embeds), move(receiver_positions), move(memory),
                        move(memory_positions), move(memory_mask), move(memory_key_limits),
                        **({"receiver_mask": move(receiver_mask)} if receiver_mask is not None else {}))
    target = output_device or args.receiver_device
    if torch.device(target) == pred.device:
        return pred
    return _SyncedTransfer.apply(pred, target)


def _renderer_inputs(args, renderer, sender_model, receiver_model, sender_cache, batch):
    memory = memory_positions = memory_mask = memory_key_limits = None
    if not args.no_sender_memory:
        memory = flatten_sender_cache(sender_model, sender_cache, batch["sender"]["position_ids"])
        memory = memory.to(args.renderer_device)
        memory_positions = batch["virtual_positions"].to(args.renderer_device)
        memory_mask = batch["sender"]["attention_mask"].to(args.renderer_device)
        if args.causal_cross_attention:
            memory_key_limits = batch["cross_causal_limits"].to(args.renderer_device)
    receiver_ids = batch["receiver_context"]["input_ids"]
    with torch.no_grad():
        embeds = receiver_model.get_input_embeddings()(receiver_ids)
    receiver_positions = batch["receiver_context"]["position_ids"]
    receiver_mask = batch["receiver_context"]["attention_mask"] if getattr(args, "token_alignment", "native") == "c2c-longest" else None
    return embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask


def kv_step(args, renderer, sender_model, receiver_model, batch):
    with torch.no_grad():
        sender_cache, target_cache = prefill_all(sender_model, receiver_model, batch)
        sender_cache = move_kv_cache(sender_cache, str(batch["sender"]["input_ids"].device))
    batch = {**batch, "receiver_context": batch["receiver"]}
    embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask = _renderer_inputs(
        args, renderer, sender_model, receiver_model, sender_cache, batch
    )
    pred = run_renderer(args, renderer, embeds, receiver_positions, memory, memory_positions,
                        memory_mask, memory_key_limits, receiver_mask, output_device=args.renderer_device)

    receiver_base = _base_model(receiver_model)
    weight = batch["receiver"]["attention_mask"][:, None, :, None].to(torch.float32).to(args.renderer_device)
    if args.arch == "dense-latent":
        weight = weight * (batch["cross_causal_limits"] > 0)[:, None, :, None].to(weight.device)
    mask_total = weight.sum().clamp_min(1.0)
    key_loss = pred.new_zeros((), dtype=torch.float32)
    value_loss = pred.new_zeros((), dtype=torch.float32)
    layer_count = pred.shape[2]
    heads, head_dim = renderer.receiver_kv_heads, renderer.receiver_head_dim
    for slot, layer_idx in enumerate(kv_layer_indices(receiver_base.config)):
        pred_key = pred[:, :, slot, 0].permute(0, 2, 1, 3).float()
        pred_value = pred[:, :, slot, 1].permute(0, 2, 1, 3).float()
        target_key, target_value = cache_layer(target_cache, layer_idx)
        length = target_key.shape[2]
        layer_weight, layer_total = weight, mask_total
        if length < pred_key.shape[2]:
            pred_key = pred_key[:, :, -length:]
            pred_value = pred_value[:, :, -length:]
            layer_weight = weight[:, :, -length:]
            layer_total = layer_weight.sum().clamp_min(1.0)
        target_key = _apply_rope(receiver_base, target_key, layer_idx, receiver_positions, inverse=True).float().to(args.renderer_device)
        target_value = target_value.float().to(args.renderer_device)
        key_scale = ((target_key.pow(2) * layer_weight).sum(dim=(0, 2, 3)) / (layer_total * head_dim)).sqrt().clamp_min(1e-3)
        value_scale = ((target_value.pow(2) * layer_weight).sum(dim=(0, 2, 3)) / (layer_total * head_dim)).sqrt().clamp_min(1e-3)
        key_scale = key_scale[None, :, None, None]
        value_scale = value_scale[None, :, None, None]
        if args.arch == "dense-latent":
            key_scale = value_scale = 1.0
        denominator = layer_total * heads * head_dim
        key_loss = key_loss + (((pred_key - target_key) / key_scale).pow(2) * layer_weight).sum() / denominator
        value_loss = value_loss + (((pred_value - target_value) / value_scale).pow(2) * layer_weight).sum() / denominator
    key_loss = key_loss / layer_count
    value_loss = value_loss / layer_count
    loss = 0.5 * (key_loss + value_loss)
    return loss, {"kv_loss": float(loss.detach()), "key_loss": float(key_loss.detach()), "value_loss": float(value_loss.detach())}


class RendererNextTokenCollator(CrossFamilyNextTokenCollator):

    def __call__(self, batch):
        rows = [row for row in (self.encode(sample) for sample in batch) if row is not None]
        rows = [row for row in rows if row["sender_ids"] and row["decode_ids"] and row["receiver_ids"]]
        if not rows:
            return None

        sender_ids = [torch.tensor(row["sender_ids"], dtype=torch.long) for row in rows]
        decode_ids = [torch.tensor(row["decode_ids"], dtype=torch.long) for row in rows]
        labels = [torch.tensor(row["labels"], dtype=torch.long) for row in rows]
        decode_positions = [
            torch.arange(row["decode_start"], row["decode_start"] + len(row["decode_ids"]), dtype=torch.long)
            for row in rows
        ]
        virtual_positions_rows = [torch.tensor(row["virtual_positions"], dtype=torch.float32) for row in rows]
        receiver_ids = [torch.tensor(row["receiver_ids"], dtype=torch.long) for row in rows]
        sender_mask_rows = [torch.tensor(row.get("sender_valid", [1] * len(row["sender_ids"])), dtype=torch.long) for row in rows]
        return {
            "sender": {
                "input_ids": _pad_stack(sender_ids, self.sender_tokenizer.pad_token_id, self.device),
                "attention_mask": _pad_stack(sender_mask_rows, 0, self.device),
                "position_ids": _pad_stack([torch.arange(ids.numel(), dtype=torch.long) for ids in sender_ids], 0, self.device),
            },
            "receiver": {
                "input_ids": _pad_stack(decode_ids, self.receiver_tokenizer.pad_token_id, self.receiver_device),
                "attention_mask": _pad_stack([torch.ones_like(ids) for ids in decode_ids], 0, self.receiver_device),
                "position_ids": _pad_stack(decode_positions, 0, self.receiver_device),
                "labels": _pad_stack(labels, -100, self.receiver_device),
            },
            "receiver_context": {
                "input_ids": _pad_stack(receiver_ids, self.receiver_tokenizer.pad_token_id, self.receiver_device),
                "attention_mask": _pad_stack([torch.tensor(row.get("receiver_valid", [1] * len(row["receiver_ids"])), dtype=torch.long) for row in rows], 0, self.receiver_device),
                "position_ids": _pad_stack([torch.arange(ids.numel(), dtype=torch.long) for ids in receiver_ids], 0, self.receiver_device),
            },
            "virtual_positions": _pad_stack(virtual_positions_rows, 0.0, self.receiver_device),
            "cross_causal_limits": _pad_stack(
                [torch.tensor(row["cross_causal_limits"], dtype=torch.long) for row in rows], 0, self.receiver_device
            ),
            "sample_weights": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32, device=self.receiver_device),
        }


class DenseLatentNextTokenCollator(RendererNextTokenCollator):

    def encode(self, sample):
        row = super().encode(sample)
        if row is not None and torch.rand(()).item() < 0.5:
            context = row["receiver_ids"]
            row["decode_ids"] = context + row["decode_ids"]
            row["labels"] = [-100] * len(context) + row["labels"]
        return row


def rendered_ce_forward(args, renderer, sender_model, receiver_model, batch):
    with torch.no_grad():
        sender_cache = sender_model(**{**batch["sender"], "use_cache": True, "past_key_values": full_sender_cache()}).past_key_values
        sender_cache = move_kv_cache(sender_cache, str(batch["sender"]["input_ids"].device))
    embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask = _renderer_inputs(
        args, renderer, sender_model, receiver_model, sender_cache, batch
    )
    pred = run_renderer(args, renderer, embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask)
    model_dtype = next(receiver_model.parameters()).dtype
    pred_cache = pred_to_cache(receiver_model, pred, receiver_positions, model_dtype)
    cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)
    receiver_inputs = batch["receiver"]
    cache_mask = batch["receiver_context"]["attention_mask"]
    from contextlib import nullcontext
    context = nullcontext()
    if args.arch == "dense-latent" or getattr(args, "checkpoint_receiver", False):
        from multiagent.dense_latent_baseline import checkpoint_receiver
        context = checkpoint_receiver(receiver_model)
    with context:
        output = receiver_model(
            input_ids=receiver_inputs["input_ids"],
            attention_mask=torch.cat([cache_mask, receiver_inputs["attention_mask"]], dim=1),
            position_ids=receiver_inputs["position_ids"],
            past_key_values=cache,
            use_cache=False,
        )
    return output.logits, receiver_inputs["labels"]


def ce_step(args, renderer, sender_model, receiver_model, batch):
    logits, labels = rendered_ce_forward(args, renderer, sender_model, receiver_model, batch)
    token_mask = labels.ne(-100)
    token_loss = _chunked_token_nll(logits, labels)

    sample_weights = batch["sample_weights"].to(device=labels.device, dtype=token_loss.dtype).unsqueeze(1)
    weighted_mask = token_mask.to(token_loss.dtype) * sample_weights
    denominator = weighted_mask.sum().clamp_min(1.0)
    loss = (token_loss * weighted_mask).sum() / denominator
    token_acc = float((logits.argmax(dim=-1).eq(labels) & token_mask).sum()) / max(int(token_mask.sum()), 1)
    hard_ce = float(loss.detach())
    return loss, {"ce_loss": hard_ce, "token_acc": token_acc, "hard_ce": hard_ce}


def _chunked_token_nll(logits, labels, chunk: int = 1024):
    pieces = []
    for start in range(0, labels.shape[1], chunk):
        piece_logits = logits[:, start : start + chunk].float()
        piece_labels = labels[:, start : start + chunk]
        pieces.append(
            F.cross_entropy(
                piece_logits.reshape(-1, piece_logits.shape[-1]), piece_labels.reshape(-1), ignore_index=-100, reduction="none"
            ).view_as(piece_labels)
        )
    return torch.cat(pieces, dim=1)


def dense_step(args, renderer, sender_model, receiver_model, batch, window: int):
    if getattr(args, "token_alignment", "native") == "c2c-longest":
        valid = batch["receiver"]["attention_mask"].bool()
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("C2C dense objective requires raw text without internal template padding")
    with torch.no_grad():
        sender_cache = sender_model(**{**batch["sender"], "use_cache": True, "past_key_values": full_sender_cache()}).past_key_values
        sender_cache = move_kv_cache(sender_cache, str(batch["sender"]["input_ids"].device))
    batch = {**batch, "receiver_context": batch["receiver"]}
    embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask = _renderer_inputs(
        args, renderer, sender_model, receiver_model, sender_cache, batch
    )
    pred = run_renderer(args, renderer, embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask)
    model_dtype = next(receiver_model.parameters()).dtype
    rendered_layers = pred_to_cache(receiver_model, pred, receiver_positions, model_dtype)

    receiver = batch["receiver"]
    attention_mask = receiver["attention_mask"]
    labels = receiver["input_ids"].masked_fill(attention_mask == 0, -100)
    labels = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1)
    lengths = attention_mask.sum(dim=1)
    lengths = None if bool((lengths == attention_mask.shape[1]).all()) else lengths

    with torch.no_grad():
        teacher_logits = receiver_model(
            input_ids=receiver["input_ids"],
            attention_mask=attention_mask,
            position_ids=receiver["position_ids"],
            use_cache=False,
        ).logits
        teacher_token_nll = _chunked_token_nll(teacher_logits, labels)
        del teacher_logits

    logits = dense_prefill_logits(
        receiver_model, rendered_layers, receiver["input_ids"], receiver["position_ids"], window, lengths
    )
    token_mask = labels.ne(-100)
    denominator = token_mask.sum().clamp_min(1)

    token_loss = _chunked_token_nll(logits, labels, chunk=512)
    mask_f = token_mask.to(token_loss.dtype)
    loss = (token_loss * mask_f).sum() / denominator
    hard_ce = float(loss.detach())
    token_acc = float((logits.argmax(dim=-1).eq(labels) & token_mask).sum()) / max(int(token_mask.sum()), 1)
    stats = {"dense_ce": float(loss.detach()), "dense_hard_ce": hard_ce, "dense_token_acc": token_acc, "dense_w": window}
    teacher_nll = float((teacher_token_nll * mask_f).sum() / denominator)
    stats["dense_teacher_nll"] = teacher_nll
    stats["dense_excess"] = hard_ce - teacher_nll
    return loss, stats


@torch.no_grad()
def downstream_generate_one(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, sample):
    context = encode_context_pair(sender_tokenizer, receiver_tokenizer, sample.sender_messages, args.max_length, args.position_mode, token_alignment=args.token_alignment)
    if not context["sender_ids"] or not context["receiver_ids"]:
        return None
    sender_ids = torch.tensor([context["sender_ids"]], dtype=torch.long, device=args.device)
    sender_valid = torch.tensor([context.get("sender_valid", [1] * len(context["sender_ids"]))], device=args.device)
    receiver_valid = torch.tensor([context.get("receiver_valid", [1] * len(context["receiver_ids"]))], device=args.receiver_device)
    sender_positions = torch.arange(sender_ids.shape[1], device=args.device).unsqueeze(0)
    sender_cache = sender_model(
        input_ids=sender_ids, attention_mask=sender_valid, position_ids=sender_positions, use_cache=True,
        past_key_values=full_sender_cache(),
    ).past_key_values
    sender_cache = move_kv_cache(sender_cache, args.device)
    receiver_ids = torch.tensor([context["receiver_ids"]], dtype=torch.long, device=args.receiver_device)
    batch = {
        "sender": {"position_ids": sender_positions, "attention_mask": sender_valid},
        "virtual_positions": torch.from_numpy(context["virtual_positions"][None].astype("float32")).to(args.receiver_device),
        "cross_causal_limits": torch.from_numpy(context["cross_causal_limits"][None]).to(args.receiver_device),
        "receiver_context": {
            "input_ids": receiver_ids,
            "attention_mask": receiver_valid,
            "position_ids": torch.arange(receiver_ids.shape[1], device=args.receiver_device).unsqueeze(0),
        },
    }
    embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask = _renderer_inputs(
        args, renderer, sender_model, receiver_model, sender_cache, batch
    )
    pred = run_renderer(args, renderer, embeds, receiver_positions, memory, memory_positions, memory_mask, memory_key_limits, receiver_mask)
    model_dtype = next(receiver_model.parameters()).dtype
    pred_cache = pred_to_cache(receiver_model, pred, receiver_positions, model_dtype)
    cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)

    if sample.prompt_mode == "split_prompt":
        prefix_ids = chat_prompt_ids(receiver_tokenizer, sample.receiver_messages, args.max_length, add_generation_prompt=True)
    else:
        prefix_ids = generation_prompt_ids(receiver_tokenizer)
    decode_start = receiver_ids.shape[1]
    attention_len = decode_start

    current_id = prefix_ids[-1]
    if len(prefix_ids) > 1:
        prefix = torch.tensor([prefix_ids[:-1]], dtype=torch.long, device=args.receiver_device)
        attention_len += prefix.shape[1]
        cache = receiver_model(
            input_ids=prefix,
            attention_mask=torch.cat([receiver_valid, receiver_valid.new_ones((1, attention_len - decode_start))], dim=1),
            position_ids=torch.arange(decode_start, decode_start + prefix.shape[1], device=args.receiver_device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        ).past_key_values

    output_ids = []
    stop_ids = eos_ids(receiver_tokenizer)
    next_position = decode_start + len(prefix_ids) - 1
    for _ in range(args.downstream_max_new_tokens):
        attention_len += 1
        output = receiver_model(
            input_ids=torch.tensor([[current_id]], dtype=torch.long, device=args.receiver_device),
            attention_mask=torch.cat([receiver_valid, receiver_valid.new_ones((1, attention_len - decode_start))], dim=1),
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=args.receiver_device),
            past_key_values=cache,
            use_cache=True,
        )
        next_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        if next_id in stop_ids:
            break
        output_ids.append(next_id)
        current_id = next_id
        cache = output.past_key_values
        next_position += 1
    return receiver_tokenizer.decode(output_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def run_downstream_eval(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, eval_setup) -> dict:
    was_training = renderer.training
    renderer.eval()
    started = time.time()
    metrics: dict[str, float | None] = {}
    try:
        total_nll = total_correct = total_count = 0
        for task, samples in eval_setup["tasks"]:
            nll_sum = token_correct = token_count = 0
            for start in range(0, len(samples), 1):
                batch = eval_setup["collator"](samples[start : start + 1])
                if batch is None:
                    continue
                logits, labels = rendered_ce_forward(args, renderer, sender_model, receiver_model, batch)
                logits = logits.float()
                mask = labels.ne(-100)
                nll_sum += float(
                    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")
                )
                token_correct += int((logits.argmax(dim=-1).eq(labels) & mask).sum())
                token_count += int(mask.sum())
            total_nll, total_correct, total_count = total_nll + nll_sum, total_correct + token_correct, total_count + token_count
            if token_count:
                metrics[f"downstream_nll:{task.spec}"] = nll_sum / token_count
                metrics[f"downstream_token_acc:{task.spec}"] = token_correct / token_count
        metrics["downstream_nll"] = total_nll / total_count if total_count else None
        metrics["downstream_token_acc"] = total_correct / total_count if total_count else None

        scores = []
        for task, samples in eval_setup["tasks"]:
            task_scores = []
            for sample in samples[: args.downstream_generate_samples]:
                text = downstream_generate_one(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, sample)
                if text is None:
                    continue
                task_scores.append(metric_from_grade(task.grade(text, sample)))
            if task_scores:
                metrics[f"downstream_score:{task.spec}"] = sum(task_scores) / len(task_scores)
                scores.extend(task_scores)
        metrics["downstream_score"] = sum(scores) / len(scores) if scores else None
        metrics["downstream_eval_seconds"] = time.time() - started
    finally:
        if was_training:
            renderer.train()
    return metrics


@torch.no_grad()
def calibrate_cache_stats(args, renderer, receiver_model, dataset, kv_collator):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=kv_collator, pin_memory=False)
    layers, heads = renderer.receiver_layers, renderer.receiver_kv_heads
    key_sums = torch.zeros(layers, heads, dtype=torch.float64)
    value_sums = torch.zeros(layers, heads, dtype=torch.float64)
    elements = torch.zeros(layers, dtype=torch.float64)
    seen = 0
    for batch in loader:
        if batch is None:
            continue
        cache = receiver_model(**{**batch["receiver"], "use_cache": True}).past_key_values
        mask = batch["receiver"]["attention_mask"][:, None, :, None].to(torch.float32)
        for slot, layer_idx in enumerate(kv_layer_indices(receiver_model.config)):
            key, value = cache_layer(cache, layer_idx)
            layer_mask = mask[:, :, -key.shape[2]:]
            key_sums[slot] += (key.float().pow(2) * layer_mask).sum(dim=(0, 2, 3)).double().cpu()
            value_sums[slot] += (value.float().pow(2) * layer_mask).sum(dim=(0, 2, 3)).double().cpu()
            elements[slot] += float(layer_mask.sum()) * renderer.receiver_head_dim
        seen += 1
        if seen >= args.stat_calibration_batches:
            break
    if seen == 0 or float(elements.sum()) == 0:
        raise RuntimeError("cache-stat calibration saw no usable batches")
    key_rms = (key_sums / elements.clamp_min(1.0)[:, None]).sqrt().clamp_min(1e-3).float()
    value_rms = (value_sums / elements.clamp_min(1.0)[:, None]).sqrt().clamp_min(1e-3).float()
    renderer.load_stat_buffers(key_rms, value_rms)
    print(
        f"cache-stat calibration ({seen} batches): "
        f"K rms mean={float(key_rms.mean()):.3g} [{float(key_rms.min()):.3g}, {float(key_rms.max()):.3g}] "
        f"V rms mean={float(value_rms.mean()):.3g} [{float(value_rms.min()):.3g}, {float(value_rms.max()):.3g}]"
    )


def _filler_id(tokenizer) -> int:
    token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = tokenizer.pad_token_id or 0
    if isinstance(token_id, list):
        token_id = token_id[0]
    return int(token_id)


def run_oom_preflight(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, optimizer):
    verify = getattr(args, "preflight_only", False)
    c2c = getattr(args, "token_alignment", "native") == "c2c-longest"
    devices = sorted({args.device, args.receiver_device, args.renderer_device})
    results = []
    current_label = None

    def begin(label):
        nonlocal current_label
        current_label = label
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        print(f"oom preflight: {label}", flush=True)

    def backward_check(loss):
        if verify and not bool(torch.isfinite(loss)):
            raise RuntimeError("preflight loss is non-finite")
        loss.backward()
        if verify:
            grads = [p.grad for p in renderer.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError("preflight gradients are missing or non-finite")
            grad_norm = torch.nn.utils.clip_grad_norm_(renderer.parameters(), args.max_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError("preflight gradient norm is non-finite")
            optimizer.step()
            memory = {device: {"peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                               "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3)}
                      for device in devices}
            print(f"preflight verified: loss={float(loss.detach()):.6g} grad_norm={float(grad_norm):.6g} memory={memory}", flush=True)
            results.append({"batch": current_label, "batch_size": args.batch_size,
                            "loss": float(loss.detach()), "grad_norm": float(grad_norm), "memory": memory})
        optimizer.zero_grad(set_to_none=True)

    def padded_mask(ids, index):
        mask = torch.ones_like(ids)
        if c2c and ids.shape[1] > index:
            mask[:, index] = 0
        return mask

    length = args.max_length
    batch_size = args.batch_size

    def positions(start, end, device):
        return torch.arange(start, end, device=device).unsqueeze(0).expand(batch_size, -1)

    sender_fill = _filler_id(sender_tokenizer)
    receiver_fill = _filler_id(receiver_tokenizer)
    sender_ids = torch.full((batch_size, length), sender_fill, dtype=torch.long, device=args.device)
    receiver_ids = torch.full((batch_size, length), receiver_fill, dtype=torch.long, device=args.receiver_device)
    common = {
        "sender": {
            "input_ids": sender_ids,
            "attention_mask": padded_mask(sender_ids, 2),
            "position_ids": positions(0, length, args.device),
        },
        "virtual_positions": positions(0, length, args.receiver_device).float(),
        "cross_causal_limits": positions(1, length + 1, args.receiver_device),
    }
    receiver_common = {
        "input_ids": receiver_ids,
        "attention_mask": padded_mask(receiver_ids, 3),
        "position_ids": positions(0, length, args.receiver_device),
    }
    ce_only = args.phase_schedule == "mixed" and args.ce_batch_fraction == 1.0
    if not ce_only:
        begin(f"kv batch at length {length}")
        loss, _ = kv_step(args, renderer, sender_model, receiver_model, {**common, "receiver": receiver_common})
        backward_check(loss)

    ce_shapes = [(length, min(args.max_trace_length, 512))]
    if args.max_trace_length > 512:
        ce_shapes.append((min(length, 4096), args.max_trace_length))
    prefix_length = length
    if args.phase_schedule != "kv-only":
        prefix_length = max(length, len(generation_prompt_ids(receiver_tokenizer)))
    ce_shapes.append((length, prefix_length + args.max_trace_length))
    if args.arch == "dense-latent":
        ce_shapes.append((length, length + prefix_length + args.max_trace_length))
    budget = getattr(args, "preflight_max_receiver_tokens", None)
    if budget:
        ce_shapes = [(length, min(args.max_trace_length, 512)),
                     (min(length, budget - args.max_trace_length), args.max_trace_length),
                     (length, budget - length)]
        ce_shapes = [(c, t) for c, t in ce_shapes if c > 0 and t > 0]
    if args.phase_schedule == "kv-only":
        ce_shapes = []
    for context_length, trace in dict.fromkeys(ce_shapes):
        ce_sender_ids = sender_ids[:, :context_length]
        ce_receiver_ids = receiver_ids[:, :context_length]
        ce_common = {
            "sender": {
                "input_ids": ce_sender_ids,
                "attention_mask": padded_mask(ce_sender_ids, 2),
                "position_ids": positions(0, context_length, args.device),
            },
            "virtual_positions": positions(0, context_length, args.receiver_device).float(),
            "cross_causal_limits": positions(1, context_length + 1, args.receiver_device),
        }
        ce_receiver_context = {
            "input_ids": ce_receiver_ids,
            "attention_mask": padded_mask(ce_receiver_ids, 3),
            "position_ids": positions(0, context_length, args.receiver_device),
        }
        decode_ids = torch.full((batch_size, trace), receiver_fill, dtype=torch.long, device=args.receiver_device)
        ce_batch = {
            **ce_common,
            "receiver_context": ce_receiver_context,
            "receiver": {
                "input_ids": decode_ids,
                "attention_mask": torch.ones_like(decode_ids),
                "position_ids": positions(context_length, context_length + trace, args.receiver_device),
                "labels": decode_ids.clone(),
            },
            "sample_weights": torch.ones(batch_size, device=args.receiver_device),
        }
        begin(f"ce batch at length {context_length}+{trace}")
        loss, _ = ce_step(args, renderer, sender_model, receiver_model, ce_batch)
        backward_check(loss)

    if args.dense_batch_fraction > 0:
        dense_length = args.dense_max_length
        dense_sender_ids = torch.full((batch_size, dense_length), sender_fill, dtype=torch.long, device=args.device)
        dense_receiver_ids = torch.full((batch_size, dense_length), receiver_fill, dtype=torch.long, device=args.receiver_device)
        dense_batch = {
            "sender": {
                "input_ids": dense_sender_ids,
                "attention_mask": torch.ones_like(dense_sender_ids),
                "position_ids": positions(0, dense_length, args.device),
            },
            "virtual_positions": positions(0, dense_length, args.receiver_device).float(),
            "cross_causal_limits": positions(1, dense_length + 1, args.receiver_device),
            "receiver": {
                "input_ids": dense_receiver_ids,
                "attention_mask": torch.ones_like(dense_receiver_ids),
                "position_ids": positions(0, dense_length, args.receiver_device),
            },
        }
        for window in (args.dense_w_choices if verify else [max(args.dense_w_choices)]):
            begin(f"dense batch at length {dense_length}, window {window}")
            loss, _ = dense_step(args, renderer, sender_model, receiver_model, dense_batch, window)
            backward_check(loss)
    if verify and c2c:
        from multiagent.dataset.c2c_types import C2CExample
        messages = [{"role": "user", "content": "The blue box contains three red balls. Café αβ."}]
        sample = C2CExample("c2c-preflight", "synthetic", "train", messages, messages,
                            [{"role": "user", "content": "How many balls are in the box?"}],
                            "Three.", "split_prompt")
        kv_collator = CrossFamilyPhase1Collator(sender_tokenizer, receiver_tokenizer,
                                                args.max_length, args.device, args.receiver_device,
                                                include_repeat=False, token_alignment="c2c-longest")
        ce_collator = RendererNextTokenCollator(sender_tokenizer, receiver_tokenizer,
                                                args.max_length, args.max_trace_length,
                                                args.device, args.receiver_device,
                                                token_alignment="c2c-longest")
        begin("real tokenizer chat KV and CE smoke")
        batch = kv_collator([sample])
        print("preflight chat padding:", {side: int((batch[side]["attention_mask"] == 0).sum())
                                          for side in ("sender", "receiver")}, flush=True)
        loss, _ = kv_step(args, renderer, sender_model, receiver_model, batch)
        backward_check(loss)
        loss, _ = ce_step(args, renderer, sender_model, receiver_model, ce_collator([sample]))
        backward_check(loss)
        downstream_generate_one(args, renderer, sender_model, receiver_model,
                                 sender_tokenizer, receiver_tokenizer, sample)
        print("preflight chat generation: ok", flush=True)
    torch.cuda.empty_cache()
    print("oom preflight: ok", flush=True)
    return results


def save_checkpoint(output_dir: Path, name: str, renderer, optimizer, scheduler, step: int, epoch: int, losses, args):
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_renderer_checkpoint(
        checkpoint_dir / "renderer.pt",
        renderer,
        {
            "phase": "cache_renderer",
            "step": step,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "sender_model": args.sender_model,
            "receiver_model": args.receiver_model,
            "position_mode": args.position_mode,
            "token_alignment": alignment_config(args.token_alignment, args.sender_model, args.receiver_model),
            "no_sender_memory": args.no_sender_memory,
        },
    )
    save_losses_and_plot(output_dir, losses)
    save_json(output_dir / "latest.json", {"checkpoint": name, "step": step, "epoch": epoch, "time": time.time()})


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    for source in (args.init_weights_from, args.resume_from):
        if source:
            payload = torch.load(Path(source) / "renderer.pt", map_location="cpu", weights_only=False)
            if checkpoint_alignment(payload) != args.token_alignment:
                raise ValueError(f"token alignment differs from checkpoint {source}")
            if args.token_alignment != "native" and (payload["sender_model"] != args.sender_model or payload["receiver_model"] != args.receiver_model):
                raise ValueError("C2C checkpoint model/tokenizer pair differs from this run")
            del payload
    torch.manual_seed(args.seed)
    from multiagent.model.vanilla_loader import resolve_model_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "args.json", vars(args))

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    sender_path = resolve_model_path(args.sender_model)
    receiver_path = resolve_model_path(args.receiver_model)
    sender_tokenizer = load_tokenizer(sender_path)
    receiver_tokenizer = load_tokenizer(receiver_path)
    receiver_tokenizer.truncation_side = "left"
    sender_model = load_model(sender_path, dtype, args.device)
    receiver_model = load_model(receiver_path, dtype, args.receiver_device)
    _receiver_wrapper = getattr(receiver_model, "model", receiver_model)
    for _vision_attr in ("vision_tower", "multi_modal_projector"):
        if getattr(_receiver_wrapper, _vision_attr, None) is not None:
            setattr(_receiver_wrapper, _vision_attr, None)
            torch.cuda.empty_cache()

    sender_config = sender_model.config.get_text_config()
    receiver_config = receiver_model.config.get_text_config()
    sender_kv_heads = getattr(sender_config, "num_key_value_heads", sender_config.num_attention_heads)
    receiver_kv_heads = getattr(receiver_config, "num_key_value_heads", receiver_config.num_attention_heads)
    sender_head_dim = getattr(sender_config, "head_dim", sender_config.hidden_size // sender_config.num_attention_heads)
    receiver_head_dim = getattr(receiver_config, "head_dim", receiver_config.hidden_size // receiver_config.num_attention_heads)
    sender_kv_layers = kv_layer_indices(sender_config)
    receiver_kv_layers = kv_layer_indices(receiver_config)
    sender_flat_dim = len(sender_kv_layers) * sender_kv_heads * sender_head_dim * 2

    if args.arch == "dense-latent":
        from multiagent.model.dense_latent_renderer import DenseLatentRenderer
        if args.bound_cache_stats or args.no_sender_memory or not args.causal_cross_attention:
            raise ValueError("dense-latent requires causal sender memory and no cache-stat bounding")
        renderer = DenseLatentRenderer(
            receiver_layers=len(receiver_kv_layers), receiver_kv_heads=receiver_kv_heads,
            receiver_head_dim=receiver_head_dim, sender_layers=len(sender_kv_layers),
            sender_kv_heads=sender_kv_heads, sender_head_dim=sender_head_dim,
        ).to(device=args.renderer_device, dtype=torch.float32)
    elif args.arch == "headwise":
        renderer = CacheRendererHeadwise(
            receiver_layers=len(receiver_kv_layers),
            receiver_kv_heads=receiver_kv_heads,
            receiver_head_dim=receiver_head_dim,
            sender_layers=len(sender_kv_layers),
            sender_kv_heads=sender_kv_heads,
            sender_head_dim=sender_head_dim,
            hidden_dim=args.headwise_hidden,
            bound_stats=args.bound_cache_stats,
        ).to(device=args.renderer_device, dtype=torch.float32)
    elif args.arch in ("v2", "pointwise"):
        renderer_cls = CacheRendererV2 if args.arch == "v2" else CacheRendererPointwise
        renderer = renderer_cls(
            receiver_hidden=receiver_config.hidden_size,
            receiver_layers=len(receiver_kv_layers),
            receiver_kv_heads=receiver_kv_heads,
            receiver_head_dim=receiver_head_dim,
            sender_layers=len(sender_kv_layers),
            sender_flat_dim=sender_flat_dim,
            d_model=args.d_model,
            n_blocks=args.n_layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            bound_stats=args.bound_cache_stats,
        ).to(device=args.renderer_device, dtype=torch.float32)
    else:
        if args.bound_cache_stats:
            raise SystemExit("--bound-cache-stats requires --arch v2")
        renderer = CacheRenderer(
            receiver_hidden=receiver_config.hidden_size,
            receiver_layers=len(receiver_kv_layers),
            receiver_kv_heads=receiver_kv_heads,
            receiver_head_dim=receiver_head_dim,
            sender_flat_dim=sender_flat_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
        ).to(device=args.renderer_device, dtype=torch.float32)
    renderer.bound_exact_grad = args.bound_stats_exact_grad
    print(f"renderer arch: {args.arch} bound_cache_stats={args.bound_cache_stats} exact_grad={args.bound_stats_exact_grad}")
    print(f"renderer params: {sum(p.numel() for p in renderer.parameters()):,}")
    print(f"sender_flat_dim: {sender_flat_dim} receiver kv per token: {len(receiver_kv_layers) * 2 * receiver_kv_heads * receiver_head_dim} "
          f"(kv layers: sender {len(sender_kv_layers)}/{sender_config.num_hidden_layers}, receiver {len(receiver_kv_layers)}/{receiver_config.num_hidden_layers})")

    optimizer = AdamW(renderer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = load_c2c_dataset_specs(
        args.datasets,
        include_answer_message=not args.exclude_answer_message,
        start=args.start,
        limit_per_dataset=args.limit_per_dataset,
    )
    kv_collator = CrossFamilyPhase1Collator(
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        max_length=args.max_length,
        device=args.device,
        receiver_device=args.receiver_device,
        position_mode=args.position_mode,
        token_alignment=args.token_alignment,
        include_repeat=False,
    )
    ce_collator_class = DenseLatentNextTokenCollator if args.arch == "dense-latent" else RendererNextTokenCollator
    ce_collator = ce_collator_class(
        sender_tokenizer=sender_tokenizer,
        receiver_tokenizer=receiver_tokenizer,
        max_source_length=args.max_length,
        max_trace_length=args.max_trace_length,
        device=args.device,
        receiver_device=args.receiver_device,
        position_mode=args.position_mode,
        token_alignment=args.token_alignment,
    )
    loaders = {
        "kv": DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=kv_collator, pin_memory=False),
    }
    if args.phase_schedule != "kv-only":
        loaders["ce"] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=ce_collator, pin_memory=False)
    if args.dense_batch_fraction > 0 and args.phase_schedule != "kv-only":
        dense_dataset = dataset
        if args.dense_datasets:
            dense_dataset = load_c2c_dataset_specs(
                args.dense_datasets,
                include_answer_message=not args.exclude_answer_message,
                start=args.start,
                limit_per_dataset=args.limit_per_dataset,
            )
        dense_collator = CrossFamilyPhase1Collator(
            sender_tokenizer=sender_tokenizer,
            receiver_tokenizer=receiver_tokenizer,
            max_length=args.dense_max_length,
            device=args.device,
            receiver_device=args.receiver_device,
            position_mode=args.position_mode,
            token_alignment=args.token_alignment,
            include_repeat=False,
        )
        loaders["dense"] = DataLoader(
            dense_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dense_collator, pin_memory=False
        )
    accum = max(1, int(args.grad_accum_steps))
    steps_per_epoch = -(-max(len(loader) for loader in loaders.values()) // accum)
    iterators = {name: cycler(loader) for name, loader in loaders.items()}
    lr_total_steps, lr_steps_source = resolve_total_steps(args, steps_per_epoch)
    scheduler = build_lr_scheduler(optimizer, args.lr_scheduler, args.warmup_steps, lr_total_steps)

    eval_setup = None
    if args.downstream_eval_every > 0:
        tasks = []
        strides = args.downstream_eval_strides or [1] * len(args.downstream_eval_datasets)
        for spec, stride in zip(args.downstream_eval_datasets, strides):
            task = load_c2c_task(spec)
            indices = list(range(0, len(task.dataset), stride))[: args.downstream_eval_samples]
            samples = [task.dataset[i] for i in indices]
            tasks.append((task, samples))
        eval_setup = {"tasks": tasks, "collator": ce_collator}
        print(f"downstream eval: every {args.downstream_eval_every} steps, " + ", ".join(f"{task.spec} x{len(samples)}" for task, samples in tasks))

    print(f"output_dir: {output_dir}")
    print(f"lr_scheduler: {args.lr_scheduler} warmup={args.warmup_steps} total={lr_total_steps} ({lr_steps_source})")
    print(f"steps_per_epoch: {steps_per_epoch} phase_schedule={args.phase_schedule}")

    losses: list[dict] = []
    window_names = (
        "loss", "kv_loss", "key_loss", "value_loss", "ce_loss", "token_acc", "hard_ce",
        "dense_ce", "dense_hard_ce", "dense_excess", "dense_token_acc",
    )
    windows = {name: deque(maxlen=args.loss_window) for name in window_names}
    step = 0
    current_epoch = 0
    if args.init_weights_from:
        payload = torch.load(Path(args.init_weights_from) / "renderer.pt", map_location="cpu", weights_only=False)
        renderer.load_state_dict(payload["renderer_state_dict"])
        print(f"initialized renderer weights from {args.init_weights_from} (step {payload.get('step')})")
    if args.resume_from:
        payload = torch.load(Path(args.resume_from) / "renderer.pt", map_location="cpu", weights_only=False)
        renderer.load_state_dict(payload["renderer_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        step = payload["step"]
        current_epoch = payload["epoch"]
        losses_path = output_dir / "losses.json"
        if losses_path.exists():
            with open(losses_path) as f:
                losses = json.load(f)
        print(f"resumed from {args.resume_from} at step {step}")

    if args.bound_cache_stats and not (args.init_weights_from or args.resume_from):
        calibrate_cache_stats(args, renderer, receiver_model, dataset, kv_collator)

    if args.oom_preflight and args.receiver_device.startswith("cuda"):
        run_oom_preflight(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, optimizer)
    if args.preflight_only:
        if not args.oom_preflight:
            raise ValueError("--preflight-only requires --oom-preflight")
        print("preflight-only: exiting before training", flush=True)
        return

    draw_generator = torch.Generator().manual_seed(args.seed + 1)

    def draw_ce_or_dense() -> str:
        if args.dense_batch_fraction > 0 and float(torch.rand((), generator=draw_generator)) < args.dense_batch_fraction:
            return "dense"
        return "ce"

    def draw_batch_type(completed_steps: int) -> str:
        if args.phase_schedule == "kv-only":
            return "kv"
        return draw_ce_or_dense() if float(torch.rand((), generator=draw_generator)) < args.ce_batch_fraction else "kv"

    def draw_dense_window() -> int:
        choices = args.dense_w_choices
        if len(choices) == 1:
            return int(choices[0])
        return int(choices[int(torch.randint(len(choices), (), generator=draw_generator))])

    try:
        renderer.train()
        for epoch in range(args.epochs):
            current_epoch = epoch + 1
            progress = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{args.epochs}")
            for _ in progress:
                if args.max_steps and step >= args.max_steps:
                    break
                micro_rows = []
                loss_total = 0.0
                for _micro in range(accum):
                    batch_type = draw_batch_type(step)
                    batch = next(iterators[batch_type])
                    if batch is None:
                        continue
                    if batch_type == "ce":
                        loss, stats = ce_step(args, renderer, sender_model, receiver_model, batch)
                    elif batch_type == "dense":
                        loss, stats = dense_step(args, renderer, sender_model, receiver_model, batch, draw_dense_window())
                    else:
                        loss, stats = kv_step(args, renderer, sender_model, receiver_model, batch)
                    (loss / accum).backward()
                    loss_total += float(loss.detach().cpu()) / accum
                    micro_rows.append((batch_type, stats))
                if not micro_rows:
                    continue
                if accum == 1:
                    batch_type, stats = micro_rows[0]
                    loss_value = loss_total
                else:
                    counts = {name: sum(1 for t, _ in micro_rows if t == name) for name in ("kv", "ce", "dense")}
                    batch_type = "mixed"
                    merged: dict[str, list[float]] = {}
                    for _t, micro_stats in micro_rows:
                        for key, value in micro_stats.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                merged.setdefault(key, []).append(float(value))
                    stats = {key: sum(values) / len(values) for key, values in merged.items()}
                    stats.update({f"n_{name}": count for name, count in counts.items()})
                    loss_value = loss_total
                grad_norm = torch.nn.utils.clip_grad_norm_(renderer.parameters(), args.max_grad_norm)
                skipped = args.grad_skip_threshold > 0 and (
                    not bool(torch.isfinite(grad_norm)) or float(grad_norm) > args.grad_skip_threshold
                )
                if not skipped:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                if args.empty_cache_every and step % args.empty_cache_every == 0:
                    for _dev in {args.device, args.receiver_device, args.renderer_device}:
                        if str(_dev).startswith("cuda"):
                            torch.cuda.synchronize(_dev)
                    torch.cuda.empty_cache()
                row = {
                    "step": step,
                    "epoch": epoch + 1,
                    "batch_type": batch_type,
                    "loss": float(loss_value),
                    **stats,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "skipped": bool(skipped),
                    "time": time.time(),
                }
                for name, window in windows.items():
                    if row.get(name) is not None:
                        window.append(float(row[name]))
                    if window:
                        row[f"{name}_avg_{args.loss_window}"] = sum(window) / len(window)

                if eval_setup is not None and step % args.downstream_eval_every == 0:
                    metrics = run_downstream_eval(args, renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, eval_setup)
                    row.update(metrics)
                    per_task = " ".join(
                        f"{key.removeprefix('downstream_score:')}={value:.4f}"
                        for key, value in metrics.items()
                        if key.startswith("downstream_score:") and value is not None
                    )
                    nll = metrics.get("downstream_nll")
                    print(
                        f"\ndownstream eval step {step}: "
                        f"nll={nll if nll is None else f'{nll:.4g}'} "
                        f"token_acc={metrics.get('downstream_token_acc'):.2%} "
                        + (f"[{per_task}] " if per_task else "")
                        + f"({metrics.get('downstream_eval_seconds'):.0f}s)",
                        flush=True,
                    )

                losses.append(row)
                if step % args.report_every == 0:
                    progress.set_postfix(
                        {name: f"{row.get(f'{name}_avg_{args.loss_window}', float('nan')):.4f}" for name in ("loss", "kv_loss", "ce_loss", "dense_ce", "dense_excess") if row.get(f"{name}_avg_{args.loss_window}") is not None},
                        refresh=False,
                    )
                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(output_dir, f"checkpoint-{step}", renderer, optimizer, scheduler, step, current_epoch, losses, args)
            if args.max_steps and step >= args.max_steps:
                break
    finally:
        save_checkpoint(output_dir, f"exit-checkpoint-{step}", renderer, optimizer, scheduler, step, current_epoch, losses, args)
        print(f"saved exit-checkpoint-{step}")


if __name__ == "__main__":
    main()
