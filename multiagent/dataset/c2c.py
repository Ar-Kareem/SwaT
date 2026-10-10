from __future__ import annotations

import jinja2
import torch

from multiagent.dataset.c2c_types import C2CExample


def _merge_for_strict_templates(messages):
    merged = []
    pending_system = ""
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "system":
            pending_system = f"{pending_system}\n\n{content}".strip()
            continue
        if pending_system:
            content = f"{pending_system}\n\n{content}"
            pending_system = ""
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] = f"{merged[-1]['content']}\n\n{content}"
        else:
            merged.append({"role": role, "content": content})
    if pending_system:
        merged.append({"role": "user", "content": pending_system})
    return merged


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool = True):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as e:
        print('RETURN THE EXCEPTION HANDLING CODE')
        raise e
    except jinja2.exceptions.TemplateError:
        return tokenizer.apply_chat_template(
            _merge_for_strict_templates(messages), enable_thinking=False, **kwargs
        )


def chat_prompt_ids(tokenizer, messages, max_length: int | None, add_generation_prompt: bool) -> list[int]:
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    if max_length is None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    if not ids:
        raise ValueError("prompt produced no tokens")
    return ids


TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>")


def turn_end_ids(tokenizer, generation_config=None) -> set[int]:
    ids: set[int] = set()
    for value in (tokenizer.eos_token_id, getattr(generation_config, "eos_token_id", None)):
        if value is None:
            continue
        ids.update(int(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
    unk = tokenizer.unk_token_id
    for token in TURN_END_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != unk and tid >= 0:
            ids.add(int(tid))
    return ids


def generation_prompt_ids(tokenizer) -> list[int]:
    sentinel = [{"role": "user", "content": "XQZC2CSENTINELZQX"}]
    base = apply_chat_template(tokenizer, sentinel, add_generation_prompt=False)
    with_generation = apply_chat_template(tokenizer, sentinel, add_generation_prompt=True)
    if not with_generation.startswith(base):
        raise ValueError("chat template does not extend the no-generation prompt; cannot isolate the generation prompt")
    ids = tokenizer(with_generation[len(base) :], add_special_tokens=False)["input_ids"]
    if not ids:
        sentinel_text = "XQZC2CSENTINELZQX"
        tail = base[base.index(sentinel_text) + len(sentinel_text) :]
        ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
        ends = turn_end_ids(tokenizer)
        while len(ids) > 1 and ids[0] in ends:
            ids = ids[1:]
    if not ids:
        raise ValueError("generation prompt produced no tokens")
    return ids


class C2CPhase1Collator:
    def __init__(
        self,
        sender_tokenizer,
        receiver_tokenizer=None,
        max_length: int = 512,
        device: str = "cuda",
        receiver_device: str | None = None,
        include_repeat_prefill: bool = False,
    ):
        self.sender_tokenizer = sender_tokenizer
        self.receiver_tokenizer = receiver_tokenizer or sender_tokenizer
        self.max_length = max_length
        self.device = device
        self.receiver_device = receiver_device or device
        self.include_repeat_prefill = include_repeat_prefill

        self.sender_tokenizer.pad_token = self.sender_tokenizer.pad_token or self.sender_tokenizer.eos_token
        self.receiver_tokenizer.pad_token = self.receiver_tokenizer.pad_token or self.receiver_tokenizer.eos_token
        self.repeat_instruction_ids = self.receiver_tokenizer(
            "\nRepeat the previous context.\n",
            add_special_tokens=False,
        )["input_ids"]

    def _tokenize(self, tokenizer, texts, device: str):
        out = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in out.items()}

    def _pad_repeat_prefill(self, receiver_tokens):
        input_ids = receiver_tokens["input_ids"]
        attention_mask = receiver_tokens["attention_mask"].bool()
        rows = []
        masks = []
        query_masks = []
        position_ids = []
        instruction = torch.tensor(self.repeat_instruction_ids, dtype=input_ids.dtype, device=input_ids.device)
        for row_ids, row_mask in zip(input_ids, attention_mask):
            context = row_ids[row_mask]
            repeated = torch.cat([context, instruction, context])
            query_mask = torch.zeros(repeated.numel(), dtype=torch.bool, device=input_ids.device)
            query_mask[context.numel() + instruction.numel() :] = True
            rows.append(repeated)
            masks.append(torch.ones_like(repeated))
            query_masks.append(query_mask)
            position_ids.append(torch.arange(repeated.numel(), dtype=torch.long, device=input_ids.device))

        max_len = max(row.numel() for row in rows)
        padded = torch.full((len(rows), max_len), self.receiver_tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_mask = torch.zeros((len(rows), max_len), dtype=input_ids.dtype, device=input_ids.device)
        padded_query_mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=input_ids.device)
        padded_position_ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=input_ids.device)
        for index, row in enumerate(rows):
            length = row.numel()
            padded[index, :length] = row
            padded_mask[index, :length] = masks[index]
            padded_query_mask[index, :length] = query_masks[index]
            padded_position_ids[index, :length] = position_ids[index]
        return {
            "input_ids": padded,
            "attention_mask": padded_mask,
            "position_ids": padded_position_ids,
            "query_mask": padded_query_mask,
        }

    def __call__(self, batch: list[C2CExample]):
        sender_texts = []
        receiver_texts = []
        for sample in batch:
            if not sample.phase1_messages:
                continue
            sender_texts.append(apply_chat_template(self.sender_tokenizer, sample.phase1_messages, add_generation_prompt=False))
            receiver_texts.append(apply_chat_template(self.receiver_tokenizer, sample.phase1_messages, add_generation_prompt=False))

        if not sender_texts:
            return None

        output = {
            "sender": self._tokenize(self.sender_tokenizer, sender_texts, self.device),
            "receiver": self._tokenize(self.receiver_tokenizer, receiver_texts, self.receiver_device),
        }
        if self.include_repeat_prefill:
            output["receiver_repeat"] = self._pad_repeat_prefill(output["receiver"])
        return output


def locate_article(text: str, article: str) -> tuple[int, str, str]:
    if not article:
        return -1, article, "empty"
    for variant, source in (
        (article, "exact"),
        (article.rstrip(), "rstrip"),
        (article.strip(), "strip"),
    ):
        if not variant:
            continue
        idx = text.find(variant)
        if idx >= 0:
            return idx, variant, source
    return -1, article, "missing"


def span_tag(article_start: int, article_end: int, full_len: int, source: str) -> str:
    return f"/span{int(article_start)}-{int(article_end)}of{int(full_len)}{source}"
