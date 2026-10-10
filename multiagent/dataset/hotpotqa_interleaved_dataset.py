from __future__ import annotations

import copy
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage


HOTPOTQA_INTERLEAVED_TASK = "hotpotqa_interleaved"
DEFAULT_FILENAME = "hotpotqa_interleaved_28k_validation.jsonl"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "extend_context"


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(output: str) -> str:
    output = str(output).strip()
    for marker in ("<|eot_id|>", "<|end_of_text|>"):
        output = output.split(marker)[0]
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL | re.IGNORECASE)
    output = re.sub(r"^(answer|the answer is)\s*:\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"^the answer is\s+", "", output, flags=re.IGNORECASE)
    return output.strip()


def grade_output(output_text: str, sample: dict[str, Any]) -> dict[str, float | str]:
    answer = str(sample.get("answer", ""))
    prediction = extract_answer(output_text)
    em = 1.0 if normalize_answer(prediction) == normalize_answer(answer) else 0.0
    f1 = token_f1_score(prediction, answer)
    return {"grade": "correct" if em else "incorrect", "f1": f1, "em": em}


def context_messages(context: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            "Use the following Wikipedia documents as source information for a "
            "multi-hop question that will follow. Use only the documents.\n\n"
            f"Context:\n{context}"
        ),
    }]


def question_messages(question: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            "Answer the question using the context above. Give only the concise answer "
            "and no explanation.\n\n"
            f"Question: {question.strip()}\n\nAnswer:"
        ),
    }]


def full_messages(context: str, question: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            context_messages(context)[0]["content"]
            + "\n\n"
            + question_messages(question)[0]["content"]
        ),
    }]


def _scan_index(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            entries.append({
                "offset": offset,
                "bytes": len(line),
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "variant": int(row["variant"]),
            })
    return entries


class HotpotQAInterleavedDataset(Dataset):

    def __init__(
        self,
        *,
        split: str = "validation",
        filename: str = DEFAULT_FILENAME,
        variant: int | None = 0,
        start: int = 0,
        limit: int | None = None,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        if split != "validation":
            raise KeyError("the interleaved HotpotQA artifact uses validation only")
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a basename inside the extend-context data directory")
        self.split = split
        self.variant = variant
        self.path = Path(data_dir) / filename
        if not self.path.exists():
            raise FileNotFoundError(
                f"generated HotpotQA file not found: {self.path}; build it with "
                "python scripts/download/build_hotpotqa_interleaved.py"
            )

        index_path = self.path.with_suffix(self.path.suffix + ".index.json")
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))["rows"]
        else:
            index = _scan_index(self.path)
        if variant is not None:
            index = [entry for entry in index if int(entry["variant"]) == variant]
        if start < 0:
            raise ValueError("start must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if start >= len(index):
            raise ValueError(f"start={start} is outside the {len(index)}-row dataset view")
        index = index[start:] if limit is None else index[start : start + limit]
        if not index:
            raise ValueError(
                f"no rows found for variant={variant!r}, start={start}, limit={limit} "
                f"in {self.path}"
            )
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def _read_row(self, index: int) -> dict[str, Any]:
        entry = self.index[index]
        with self.path.open("rb") as handle:
            handle.seek(int(entry["offset"]))
            payload = handle.read(int(entry["bytes"]))
        return json.loads(payload)

    def __getitem__(self, index: int) -> C2CExample:
        row = self._read_row(index)
        context = str(row["context"])
        question = str(row["question"])
        answer = str(row["answer"])
        messages = full_messages(context, question)
        metadata = {
            **row,
            "dataset": HOTPOTQA_INTERLEAVED_TASK,
            "split": self.split,
        }
        return C2CExample(
            id=str(row["id"]),
            dataset=HOTPOTQA_INTERLEAVED_TASK,
            split=self.split,
            phase1_messages=messages,
            sender_messages=context_messages(context),
            receiver_messages=question_messages(question),
            target_text=answer,
            prompt_mode="split_prompt",
            messages=messages,
            metadata=metadata,
        )

    def select(self, indices: list[int]):
        assert all(0 <= index < len(self.index) for index in indices)
        other = copy.deepcopy(self)
        other.index = [other.index[index] for index in indices]
        return other
