from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.dataset.qa_dataset import (
    DEFAULT_ROOT,
    best_subspan_em,
    extract_answer,
    squad_exact_match,
    squad_f1,
)

DATASET_DIRS = {
    "triviaqa": "mandarjoshi__trivia_qa/unfiltered.nocontext",
    "nq-open": "google-research-datasets__nq_open/default",
}
CLOSEDBOOK_TASKS = set(DATASET_DIRS)

INSTRUCTION = "Answer the following question. Only give me the answer and do not output any other words."


def closedbook_messages(question: str) -> list[ChatMessage]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"}]


def _parse(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _answers(name: str, row: dict[str, Any]) -> tuple[list[str], str]:
    raw = _parse(row["answer"])
    if name == "triviaqa":
        canonical = str(raw["value"])
        aliases = [canonical, *raw.get("aliases", []), *raw.get("normalized_aliases", [])]
    else:
        aliases = [str(a) for a in raw]
        canonical = aliases[0]
    seen, answers = set(), []
    for alias in aliases:
        alias = str(alias).strip()
        if alias and alias not in seen:
            seen.add(alias)
            answers.append(alias)
    return answers, canonical


def grade_output(output_text: str, sample: dict[str, Any]) -> dict[str, Any] | str:
    answers = [str(a) for a in sample.get("answers", [])]
    if not answers:
        return "fail"
    prediction = extract_answer(output_text).split("\n")[0].strip()
    em = squad_exact_match(prediction, answers)
    return {
        "grade": "correct" if em else "incorrect",
        "exact_match": em,
        "f1": squad_f1(prediction, answers),
        "subspan_em": float(best_subspan_em(prediction, answers)),
    }


class ClosedBookQADataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "validation",
        root: str | Path = DEFAULT_ROOT,
        include_answer_message: bool = False,
    ) -> None:
        if name not in DATASET_DIRS:
            raise KeyError(f"unknown closed-book dataset {name!r}; choose from {sorted(DATASET_DIRS)}")
        if include_answer_message and split != "train":
            raise ValueError("include_answer_message=True is only allowed for split='train'")
        path = Path(root) / "snapshots" / DATASET_DIRS[name]
        if not path.exists():
            raise FileNotFoundError(
                f"snapshot not found: {path}. Download it with: "
                f"python scripts/download/download_eval_datasets.py --dataset {name}"
            )
        dataset = load_from_disk(str(path))
        if split not in dataset:
            raise KeyError(f"split {split!r} not found for {name}; available: {list(dataset)}")
        self.name = name
        self.split = split
        self.include_answer_message = include_answer_message
        self.data = [{"question": q, "answer": a} for q, a in zip(dataset[split]["question"], dataset[split]["answer"])]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> C2CExample:
        row = self.data[index]
        question = str(row["question"]).strip()
        answers, canonical = _answers(self.name, row)
        messages = closedbook_messages(question)
        if self.include_answer_message:
            messages = [*messages, {"role": "assistant", "content": canonical}]
        item_id = f"{self.name}/{self.split}/{index}"
        metadata = {
            "id": item_id,
            "dataset": self.name,
            "split": self.split,
            "question": question,
            "answers": answers,
            "answer": canonical,
        }
        return C2CExample(
            id=item_id,
            dataset=self.name,
            split=self.split,
            phase1_messages=messages,
            sender_messages=messages,
            receiver_messages=messages,
            target_text=canonical,
            prompt_mode="same_prompt_continuation",
            messages=messages,
            metadata=metadata,
        )

    def select(self, indices: list[int]):
        assert all(0 <= i < len(self.data) for i in indices)
        other = copy.deepcopy(self)
        other.data = [other.data[i] for i in indices]
        return other
