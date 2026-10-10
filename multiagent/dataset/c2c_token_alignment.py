from __future__ import annotations

from collections import Counter
from functools import lru_cache

from multiagent.dataset.c2c import apply_chat_template

TOKEN_ALIGNMENTS = ("native", "c2c-longest")
UPSTREAM_REVISION = "04b1d5098f2d7aedd375260066e66e7f38682bb1"


def alignment_config(mode, sender=None, receiver=None):
    if mode not in TOKEN_ALIGNMENTS:
        raise ValueError(f"unknown token alignment: {mode!r}")
    config = {"mode": mode, "version": 1}
    if mode == "c2c-longest":
        config.update(strategy="longest", positions="aligned-integer-slots",
                      template_padding="right-pad-sections-mask-keys",
                      fallback="sender-unk-or-zero", tie_break="first",
                      sender_tokenizer=sender, receiver_tokenizer=receiver,
                      upstream_revision=UPSTREAM_REVISION)
    return config


def checkpoint_alignment(payload):
    config = payload.get("token_alignment", {"mode": "native", "version": 1})
    mode = config["mode"]
    if config.get("version") != 1 or mode not in TOKEN_ALIGNMENTS:
        raise ValueError(f"unsupported token-alignment metadata: {config}")
    if mode == "c2c-longest":
        expected = alignment_config(mode)
        for key in ("strategy", "positions", "template_padding", "fallback", "tie_break"):
            if config.get(key) != expected[key]:
                raise ValueError(f"unsupported token-alignment policy {key}: {config.get(key)}")
        if payload["renderer_config"].get("arch") != "v2" or not payload.get("args", {}).get("causal_cross_attention"):
            raise ValueError("C2C-longest requires v2 with causal cross-attention")
    return mode


class LongestTokenAligner:
    def __init__(self, sender, receiver):
        self.sender, self.receiver = sender, receiver
        self.sender_vocab_size = len(sender)
        self.receiver_special_ids = set(receiver.all_special_ids)
        self.cache = {}
        self.stats = Counter()

    @staticmethod
    def decode(tokenizer, token):
        return tokenizer.decode([token], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)

    def convert(self, receiver_ids):
        output = []
        for token in receiver_ids:
            token = int(token)
            if token not in self.cache:
                text = self.decode(self.receiver, token)
                fallback = False
                if token in self.receiver_special_ids:
                    special = {getattr(self.receiver, f"{kind}_token_id"):
                               getattr(self.sender, f"{kind}_token_id")
                               for kind in ("pad", "eos", "bos", "unk")}
                    selected = special.get(token)
                    if selected is None:
                        selected = self.sender.convert_tokens_to_ids(text)
                        if selected is None or selected == self.sender.unk_token_id:
                            selected, fallback = self.sender.unk_token_id or 0, True
                    candidates = [selected]
                else:
                    candidates = self.sender.encode(text, add_special_tokens=False)
                    fallback = not candidates
                    selected = max(candidates, key=lambda t: len(self.decode(self.sender, t))) if candidates else self.sender.unk_token_id or 0
                if not isinstance(selected, int) or not 0 <= selected < self.sender_vocab_size:
                    raise ValueError(f"invalid sender ID {selected!r} for receiver token {token}")
                self.cache[token] = (selected, len(candidates) > 1, fallback,
                                     self.decode(self.sender, selected) != text)
            selected, ambiguous, fallback, changed = self.cache[token]
            output.append(selected)
            self.stats.update(tokens=1, ambiguous=int(ambiguous), fallback=int(fallback), changed=int(changed))
        return output


@lru_cache(maxsize=8)
def get_aligner(sender, receiver):
    return LongestTokenAligner(sender, receiver)


def _sections(tokenizer, messages):
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    sections, previous, search_from = [], 0, 0
    for message in messages:
        content = str(message.get("content", ""))
        if not content:
            continue
        start = -1
        for candidate in dict.fromkeys((content, content.strip())):
            if not candidate:
                continue
            at = text.find(candidate, search_from)
            first = at
            while at >= 0:
                end = at + len(candidate)
                if end == len(text) or text[end] == "<":
                    break
                at = text.find(candidate, at + 1)
            start = at if at >= 0 else first
            if start >= 0:
                content = candidate
                break
        if start < 0:
            raise ValueError("message content absent from chat template; cannot align")
        end = start + len(content)
        left = previous
        while left < len(ids) and offsets[left][1] <= start:
            left += 1
        right = left
        while right < len(ids) and offsets[right][0] < end:
            right += 1
        sections.extend((("template", ids[previous:left]), ("message", ids[left:right])))
        previous, search_from = right, end
    sections.append(("template", ids[previous:]))
    return sections


def encode_c2c_context(sender, receiver, messages, max_length):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    aligner = get_aligner(sender, receiver)
    before = aligner.stats.copy()
    sender_ids, receiver_ids, sender_valid, receiver_valid = [], [], [], []
    raw = len(messages) == 1 and messages[0].get("role") == "raw"
    if raw:
        receiver_ids = receiver.encode(str(messages[0].get("content", "")), add_special_tokens=False)[:max_length]
        sender_ids = aligner.convert(receiver_ids)
        sender_valid = [1] * len(sender_ids)
        receiver_valid = [1] * len(receiver_ids)
    else:
        sender_sections, receiver_sections = _sections(sender, messages), _sections(receiver, messages)
        if len(sender_sections) != len(receiver_sections):
            raise ValueError("sender/receiver chat section count mismatch")
        for (s_kind, s_ids), (r_kind, r_ids) in zip(sender_sections, receiver_sections):
            if s_kind != r_kind:
                raise ValueError("sender/receiver chat section type mismatch")
            remaining = max_length - len(receiver_ids)
            if remaining <= 0:
                break
            if r_kind == "message":
                r_ids = r_ids[:remaining]
                s_ids = aligner.convert(r_ids)
            length = min(max(len(s_ids), len(r_ids)), remaining)
            for tokenizer, segment, output, valid in (
                (sender, s_ids, sender_ids, sender_valid),
                (receiver, r_ids, receiver_ids, receiver_valid),
            ):
                segment = segment[:length]
                pad = tokenizer.pad_token_id
                if pad is None:
                    pad = tokenizer.eos_token_id
                if len(segment) < length and pad is None:
                    raise ValueError("template alignment requires PAD or EOS")
                output.extend(segment + [pad] * (length - len(segment)))
                valid.extend([1] * len(segment) + [0] * (length - len(segment)))
    stats = dict(aligner.stats - before)
    stats.update(sender_padding=sender_valid.count(0), receiver_padding=receiver_valid.count(0),
                 slots=len(receiver_ids))
    return dict(sender_ids=sender_ids, receiver_ids=receiver_ids,
                sender_valid=sender_valid, receiver_valid=receiver_valid,
                position_ids=list(range(len(receiver_ids))), diagnostics=stats)
