from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ChatMessage = dict[str, str]
PromptMode = Literal["same_prompt_continuation", "split_prompt", "raw_continuation"]


@dataclass(frozen=True, slots=True)
class C2CExample:
    id: str
    dataset: str
    split: str
    phase1_messages: list[ChatMessage]
    sender_messages: list[ChatMessage]
    receiver_messages: list[ChatMessage]
    target_text: str | None
    prompt_mode: PromptMode
    messages: list[ChatMessage] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def c2c_example_to_dict(example: C2CExample) -> dict[str, Any]:
    row = asdict(example)
    metadata = row.pop("metadata", {})
    return {**metadata, **row}
