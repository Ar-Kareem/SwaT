from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch

from multiagent.dataset.c2c import apply_chat_template, chat_prompt_ids, generation_prompt_ids
from multiagent.dataset.c2c_types import C2CExample


REPEAT_INSTRUCTION = "\nRepeat the previous context.\n"
POSITION_MODES = ("interp", "naive", "scaled")


def require_fast(tokenizer, name: str):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{name} tokenizer must be a fast tokenizer (offset mappings required)")
    return tokenizer


def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return ids, offsets


def message_regions(sender_text: str, receiver_text: str, messages) -> list[tuple[int, int, int, int]]:
    regions = []
    sender_from = receiver_from = 0
    for message in messages:
        content = str(message.get("content", ""))
        if not content:
            continue
        sender_start = sender_text.find(content, sender_from)
        receiver_start = receiver_text.find(content, receiver_from)
        if sender_start < 0 or receiver_start < 0:
            stripped = content.strip()
            if stripped:
                sender_start = sender_text.find(stripped, sender_from)
                receiver_start = receiver_text.find(stripped, receiver_from)
                content = stripped
        if sender_start < 0 or receiver_start < 0:
            raise ValueError("message content not found verbatim in templated text; cannot align")
        regions.append((sender_start, sender_start + len(content), receiver_start, receiver_start + len(content)))
        sender_from = sender_start + len(content)
        receiver_from = receiver_start + len(content)
    return regions


def char_map_anchors(regions, sender_length: int, receiver_length: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [0.0], [0.0]
    for sender_start, sender_end, receiver_start, receiver_end in regions:
        for x, y in ((float(sender_start), float(receiver_start)), (float(sender_end), float(receiver_end))):
            if x > xs[-1] and y > ys[-1]:
                xs.append(x)
                ys.append(y)
    if sender_length > xs[-1] and receiver_length > ys[-1]:
        xs.append(float(sender_length))
        ys.append(float(receiver_length))
    return np.asarray(xs), np.asarray(ys)


def truncate_to_shared_budget(sender_ids, sender_offsets, receiver_ids, receiver_offsets, max_length: int, xs, ys):
    if len(sender_ids) <= max_length and len(receiver_ids) <= max_length:
        return sender_ids, sender_offsets, receiver_ids, receiver_offsets
    sender_char_limit = float(sender_offsets[min(max_length, len(sender_offsets)) - 1][1])
    receiver_char_limit = float(receiver_offsets[min(max_length, len(receiver_offsets)) - 1][1])
    receiver_limit_in_sender = float(np.interp(receiver_char_limit, ys, xs))
    common_sender = min(sender_char_limit, receiver_limit_in_sender)
    common_receiver = float(np.interp(common_sender, xs, ys))

    sender_ends = [end for _, end in sender_offsets]
    receiver_ends = [end for _, end in receiver_offsets]
    sender_keep = bisect_right(sender_ends, common_sender + 0.5)
    receiver_keep = bisect_right(receiver_ends, common_receiver + 0.5)
    sender_keep = max(1, min(sender_keep, max_length))
    receiver_keep = max(1, min(receiver_keep, max_length))
    return (
        sender_ids[:sender_keep],
        sender_offsets[:sender_keep],
        receiver_ids[:receiver_keep],
        receiver_offsets[:receiver_keep],
    )


def aligned_pairs(sender_offsets, receiver_offsets, regions) -> list[tuple[int, int]]:
    receiver_by_span = {span: i for i, span in enumerate(receiver_offsets) if span[1] > span[0]}
    region_starts = [region[0] for region in regions]
    pairs = []
    for j, (start, end) in enumerate(sender_offsets):
        if end <= start:
            continue
        k = bisect_right(region_starts, start) - 1
        if k < 0:
            continue
        sender_start, sender_end, receiver_start, _ = regions[k]
        if start < sender_start or end > sender_end:
            continue
        shift = receiver_start - sender_start
        i = receiver_by_span.get((start + shift, end + shift))
        if i is not None:
            pairs.append((j, i))
    return pairs


def virtual_positions(sender_offsets, receiver_offsets, xs, ys, mode: str) -> np.ndarray:
    sender_count = len(sender_offsets)
    receiver_count = len(receiver_offsets)
    if mode == "naive":
        return np.arange(sender_count, dtype=np.float64)
    if mode == "scaled":
        return np.arange(sender_count, dtype=np.float64) * (receiver_count / max(sender_count, 1))
    if mode != "interp":
        raise ValueError(f"unknown position mode {mode!r}")

    receiver_starts = [start for start, _ in receiver_offsets]
    midpoints = np.asarray([(start + end) / 2.0 for start, end in sender_offsets])
    mapped = np.interp(midpoints, xs, ys)
    positions = np.empty(sender_count, dtype=np.float64)
    last_receiver = receiver_count - 1
    for j, midpoint in enumerate(mapped):
        k = bisect_right(receiver_starts, midpoint) - 1
        k = min(max(k, 0), last_receiver)
        start, end = receiver_offsets[k]
        width = max(end - start, 1)
        fraction = min(max((midpoint - start) / width, 0.0), 1.0)
        positions[j] = max(k + fraction - 0.5, 0.0)
    return positions


def cross_causal_limits(sender_offsets, receiver_offsets, xs, ys) -> np.ndarray:
    sender_ends = np.asarray([end for _, end in sender_offsets], dtype=np.float64)
    receiver_ends = np.asarray([end for _, end in receiver_offsets], dtype=np.float64)
    mapped = np.interp(sender_ends, xs, ys)
    mapped = np.maximum.accumulate(mapped)
    return np.searchsorted(mapped, receiver_ends, side="right").astype(np.int64)


def encode_context_pair(sender_tokenizer, receiver_tokenizer, messages, max_length: int, position_mode: str, token_alignment: str = "native", sample_id: str | None = None) -> dict:
    if token_alignment == "c2c-longest":
        from multiagent.dataset.c2c_token_alignment import encode_c2c_context
        try:
            row = encode_c2c_context(sender_tokenizer, receiver_tokenizer, messages, max_length)
        except ValueError as error:
            raise ValueError(f"C2C alignment failed for {sample_id or 'context'}: {error}") from error
        slots = np.asarray(row["position_ids"], dtype=np.int64)
        return {**row, "virtual_positions": slots, "cross_causal_limits": slots + 1, "aligned": []}
    if token_alignment != "native":
        raise ValueError(f"unknown token alignment {token_alignment!r}")
    raw_mode = len(messages) == 1 and messages[0].get("role") == "raw"
    if raw_mode:
        sender_text = receiver_text = str(messages[0].get("content", ""))
    else:
        sender_text = apply_chat_template(sender_tokenizer, messages, add_generation_prompt=False)
        receiver_text = apply_chat_template(receiver_tokenizer, messages, add_generation_prompt=False)
    regions = message_regions(sender_text, receiver_text, messages)
    xs, ys = char_map_anchors(regions, len(sender_text), len(receiver_text))
    sender_ids, sender_offsets = tokenize_with_offsets(sender_tokenizer, sender_text)
    receiver_ids, receiver_offsets = tokenize_with_offsets(receiver_tokenizer, receiver_text)
    sender_ids, sender_offsets, receiver_ids, receiver_offsets = truncate_to_shared_budget(
        sender_ids, sender_offsets, receiver_ids, receiver_offsets, max_length, xs, ys
    )
    return {
        "sender_ids": sender_ids,
        "receiver_ids": receiver_ids,
        "virtual_positions": virtual_positions(sender_offsets, receiver_offsets, xs, ys, position_mode),
        "aligned": aligned_pairs(sender_offsets, receiver_offsets, regions),
        "cross_causal_limits": cross_causal_limits(sender_offsets, receiver_offsets, xs, ys),
    }


def _pad_stack(rows: list[torch.Tensor], pad_value, device: str) -> torch.Tensor:
    max_len = max(row.numel() for row in rows)
    out = torch.full((len(rows), max_len), pad_value, dtype=rows[0].dtype, device=device)
    for index, row in enumerate(rows):
        out[index, : row.numel()] = row.to(device)
    return out


class CrossFamilyPhase1Collator:
    def __init__(
        self,
        sender_tokenizer,
        receiver_tokenizer,
        max_length: int,
        device: str,
        receiver_device: str,
        position_mode: str = "interp",
        include_repeat: bool = True,
        token_alignment: str = "native",
    ):
        self.sender_tokenizer = require_fast(sender_tokenizer, "sender")
        self.receiver_tokenizer = require_fast(receiver_tokenizer, "receiver")
        self.max_length = max_length
        self.device = device
        self.receiver_device = receiver_device
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")
        self.position_mode = position_mode
        self.token_alignment = token_alignment
        if token_alignment != "native" and include_repeat:
            raise ValueError("C2C alignment is supported by the renderer without repeat inputs")
        self.include_repeat = include_repeat
        self.sender_tokenizer.pad_token = self.sender_tokenizer.pad_token or self.sender_tokenizer.eos_token
        self.receiver_tokenizer.pad_token = self.receiver_tokenizer.pad_token or self.receiver_tokenizer.eos_token
        self.repeat_instruction_ids = self.receiver_tokenizer(REPEAT_INSTRUCTION, add_special_tokens=False)["input_ids"]

    def encode(self, sample: C2CExample) -> dict | None:
        if not sample.phase1_messages:
            return None
        return encode_context_pair(
            self.sender_tokenizer,
            self.receiver_tokenizer,
            sample.phase1_messages,
            self.max_length,
            self.position_mode,
            token_alignment=self.token_alignment,
            sample_id=sample.id,
        )

    def __call__(self, batch: list[C2CExample]):
        rows = []
        for sample in batch:
            row = self.encode(sample)
            if row is not None and row["sender_ids"] and row["receiver_ids"]:
                rows.append(row)
        if not rows:
            return None

        sender_ids = [torch.tensor(row["sender_ids"], dtype=torch.long) for row in rows]
        receiver_ids = [torch.tensor(row["receiver_ids"], dtype=torch.long) for row in rows]
        virtual_positions_rows = [torch.tensor(row["virtual_positions"], dtype=torch.float32) for row in rows]
        align_sender = [torch.tensor([j for j, _ in row["aligned"]], dtype=torch.long) for row in rows]
        align_receiver = [torch.tensor([i for _, i in row["aligned"]], dtype=torch.long) for row in rows]
        align_masks = [torch.ones(len(row["aligned"]), dtype=torch.bool) for row in rows]

        sender_input_ids = _pad_stack(sender_ids, self.sender_tokenizer.pad_token_id, self.device)
        sender_mask = _pad_stack([torch.tensor(row.get("sender_valid", [1] * len(row["sender_ids"])), dtype=torch.long) for row in rows], 0, self.device)
        sender_positions = _pad_stack([torch.arange(ids.numel(), dtype=torch.long) for ids in sender_ids], 0, self.device)

        receiver_input_ids = _pad_stack(receiver_ids, self.receiver_tokenizer.pad_token_id, self.receiver_device)
        receiver_mask = _pad_stack([torch.tensor(row.get("receiver_valid", [1] * len(row["receiver_ids"])), dtype=torch.long) for row in rows], 0, self.receiver_device)
        receiver_positions = _pad_stack([torch.arange(ids.numel(), dtype=torch.long) for ids in receiver_ids], 0, self.receiver_device)

        has_alignment = any(mask.numel() for mask in align_masks)
        if not has_alignment:
            align_sender = [torch.zeros(1, dtype=torch.long) for _ in rows]
            align_receiver = [torch.zeros(1, dtype=torch.long) for _ in rows]
            align_masks = [torch.zeros(1, dtype=torch.bool) for _ in rows]
        else:
            align_sender = [row if row.numel() else torch.zeros(1, dtype=torch.long) for row in align_sender]
            align_receiver = [row if row.numel() else torch.zeros(1, dtype=torch.long) for row in align_receiver]
            align_masks = [
                row if row.numel() else torch.zeros(1, dtype=torch.bool)
                for row in align_masks
            ]

        output = {
            "sender": {
                "input_ids": sender_input_ids,
                "attention_mask": sender_mask,
                "position_ids": sender_positions,
            },
            "receiver": {
                "input_ids": receiver_input_ids,
                "attention_mask": receiver_mask,
                "position_ids": receiver_positions,
            },
            "virtual_positions": _pad_stack(virtual_positions_rows, 0.0, self.receiver_device),
            "cross_causal_limits": _pad_stack(
                [torch.tensor(row["cross_causal_limits"], dtype=torch.long) for row in rows], 0, self.receiver_device
            ),
            "align_sender": _pad_stack(align_sender, 0, self.receiver_device),
            "align_receiver": _pad_stack(align_receiver, 0, self.receiver_device),
            "align_mask": _pad_stack(align_masks, False, self.receiver_device),
            "aligned_tokens": int(sum(len(row["aligned"]) for row in rows)),
            "sender_tokens": int(sum(len(row["sender_ids"]) for row in rows)),
        }
        if self.include_repeat:
            output["receiver_repeat"] = self._repeat_inputs(receiver_ids)
        return output

    def _repeat_inputs(self, receiver_ids: list[torch.Tensor]) -> dict:
        instruction = torch.tensor(self.repeat_instruction_ids, dtype=torch.long)
        rows, masks, query_masks, position_rows = [], [], [], []
        for context in receiver_ids:
            repeated = torch.cat([context, instruction, context])
            query_mask = torch.zeros(repeated.numel(), dtype=torch.bool)
            query_mask[context.numel() + instruction.numel() :] = True
            rows.append(repeated)
            masks.append(torch.ones_like(repeated))
            query_masks.append(query_mask)
            position_rows.append(torch.arange(repeated.numel(), dtype=torch.long))
        return {
            "input_ids": _pad_stack(rows, self.receiver_tokenizer.pad_token_id, self.receiver_device),
            "attention_mask": _pad_stack(masks, 0, self.receiver_device),
            "position_ids": _pad_stack(position_rows, 0, self.receiver_device),
            "query_mask": _pad_stack(query_masks, False, self.receiver_device),
        }


class CrossFamilyNextTokenCollator:

    def __init__(
        self,
        sender_tokenizer,
        receiver_tokenizer,
        max_source_length: int,
        max_trace_length: int,
        device: str,
        receiver_device: str,
        position_mode: str = "interp",
        append_eos: bool = True,
        include_teacher: bool = False,
        token_alignment: str = "native",
    ):
        self.sender_tokenizer = require_fast(sender_tokenizer, "sender")
        self.receiver_tokenizer = require_fast(receiver_tokenizer, "receiver")
        self.max_source_length = max_source_length
        self.max_trace_length = max_trace_length
        self.device = device
        self.receiver_device = receiver_device
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")
        self.position_mode = position_mode
        self.token_alignment = token_alignment
        self.append_eos = append_eos
        if token_alignment != "native" and include_teacher:
            raise ValueError("C2C alignment uses RendererNextTokenCollator, without teacher collation")
        self.include_teacher = include_teacher
        self.sender_tokenizer.pad_token = self.sender_tokenizer.pad_token or self.sender_tokenizer.eos_token
        self.receiver_tokenizer.pad_token = self.receiver_tokenizer.pad_token or self.receiver_tokenizer.eos_token

    def encode(self, sample: C2CExample) -> dict | None:
        if sample.target_text is None or not sample.sender_messages:
            return None
        context = encode_context_pair(
            self.sender_tokenizer,
            self.receiver_tokenizer,
            sample.sender_messages,
            self.max_source_length,
            self.position_mode,
            token_alignment=self.token_alignment,
            sample_id=sample.id,
        )
        if sample.prompt_mode == "split_prompt":
            prefix_ids = chat_prompt_ids(self.receiver_tokenizer, sample.receiver_messages, self.max_source_length, add_generation_prompt=True)
        elif sample.prompt_mode == "same_prompt_continuation":
            prefix_ids = generation_prompt_ids(self.receiver_tokenizer)
        elif sample.prompt_mode == "raw_continuation":
            prefix_ids = []
        else:
            raise ValueError(f"unknown prompt_mode {sample.prompt_mode!r}")
        trace_ids = self.receiver_tokenizer(str(sample.target_text), add_special_tokens=False)["input_ids"]
        if sample.prompt_mode != "raw_continuation" and self.append_eos and self.receiver_tokenizer.eos_token_id is not None:
            trace_ids = [*trace_ids, int(self.receiver_tokenizer.eos_token_id)]
        trace_ids = trace_ids[: self.max_trace_length]
        if sample.prompt_mode == "raw_continuation":
            if len(trace_ids) < 2:
                return None
            decode_ids = trace_ids[:-1]
            labels = trace_ids[1:]
        else:
            decode_ids = [*prefix_ids, *trace_ids[:-1]]
            labels = [-100] * (len(prefix_ids) - 1) + trace_ids
        if not decode_ids:
            return None
        return {
            **{key: context[key] for key in ("sender_valid", "receiver_valid", "diagnostics") if key in context},
            "sender_ids": context["sender_ids"],
            "receiver_ids": context["receiver_ids"],
            "virtual_positions": context["virtual_positions"],
            "cross_causal_limits": context["cross_causal_limits"],
            "decode_start": len(context["receiver_ids"]),
            "decode_ids": decode_ids,
            "labels": labels,
            "sample_weight": float(sample.metadata.get("phase2_loss_weight", 1.0)),
        }

    def __call__(self, batch: list[C2CExample]):
        rows = [row for row in (self.encode(sample) for sample in batch) if row is not None]
        rows = [row for row in rows if row["sender_ids"] and row["decode_ids"]]
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

        sender_mask_rows = [torch.tensor(row.get("sender_valid", [1] * len(row["sender_ids"])), dtype=torch.long) for row in rows]
        teacher = None
        if self.include_teacher:
            teacher_ids = [
                torch.tensor([*row["receiver_ids"], *row["decode_ids"]], dtype=torch.long) for row in rows
            ]
            teacher = {
                "input_ids": _pad_stack(teacher_ids, self.receiver_tokenizer.pad_token_id, self.receiver_device),
                "attention_mask": _pad_stack([torch.ones_like(ids) for ids in teacher_ids], 0, self.receiver_device),
            }
        return {
            **({"teacher": teacher} if teacher is not None else {}),
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
            "cache_attention_mask": _pad_stack(sender_mask_rows, 0, self.receiver_device),
            "virtual_positions": _pad_stack(virtual_positions_rows, 0.0, self.receiver_device),
            "sample_weights": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32, device=self.receiver_device),
        }
