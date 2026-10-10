from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from datasets import Dataset as DatasetsDataset
from datasets import load_from_disk
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.paths import DATASETS_ROOT


DEFAULT_ROOT = DATASETS_ROOT

DATASET_DIRS = {
    "gsm8k": ["openai__gsm8k/main", "openai__gsm8k/default"],
    "math-500": ["HuggingFaceH4__MATH-500/default"],
    "math-algebra": [
        "EleutherAI__hendrycks_math/algebra",
        "hendrycks__competition_math/algebra",
        "hendrycks_math/algebra",
    ],
}

def _strip_answer(value: Any) -> str:
    return str(value).strip()


def _last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_gsm8k_answer(solution: Any) -> str | None:
    text = _strip_answer(solution)
    if "####" in text:
        return _last_number(text.split("####")[-1])
    return _last_number(text)


def extract_boxed_answer(text: Any) -> str | None:
    text = _strip_answer(text)
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None

    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return None


def extract_math_answer(text: Any) -> str | None:
    text = _strip_answer(text)
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    match = re.search(r"(?i)(?:final answer|answer)\s*[:=]\s*([^\n]+)", text)
    if match:
        return match.group(1).strip().rstrip(".")

    return _last_number(text)


def _normalize_simple_answer(value: Any) -> str:
    text = _strip_answer(value)
    text = re.sub(r"^\\boxed\s*\{(.*)\}$", r"\1", text)
    text = text.strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    while text.endswith(")") and text.count(")") > text.count("("):
        text = text[:-1]
    while text.endswith("]") and text.count("]") > text.count("["):
        text = text[:-1]
    return text


def _math_verify_equal(prediction: str, answer: str) -> bool | None:
    gold_text = answer if answer.startswith("$") else f"${answer}$"
    gold = parse(gold_text, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    pred = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )

    if not gold or not pred:
        return None
    return bool(verify(gold, pred))


def grade_output(output_text: str, sample: dict[str, Any]) -> str:
    if sample["answer_type"] == "numeric":
        prediction = extract_gsm8k_answer(output_text)
        if prediction is None:
            return "fail"
        return "correct" if _normalize_simple_answer(prediction) == _normalize_simple_answer(sample["answer"]) else "incorrect"

    prediction = extract_math_answer(output_text)
    if prediction is None:
        return "fail"

    if _normalize_simple_answer(prediction) == _normalize_simple_answer(sample["answer"]):
        return "correct"

    verified = _math_verify_equal(output_text, sample["answer"])
    if verified:
        return "correct"
    for candidate in _fallback_answer_candidates(output_text):
        latex = _unicode_math_to_latex(candidate)
        if _normalize_simple_answer(latex) == _normalize_simple_answer(sample["answer"]):
            return "correct"
        if _math_verify_equal(output_text + "\n\\boxed{" + latex + "}", sample["answer"]):
            return "correct"
    return "incorrect"


_SUPERSCRIPTS = {"⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9"}


def _unicode_math_to_latex(answer: str) -> str:
    text = answer.strip().strip("*").strip().rstrip(".").strip()
    text = re.sub(r"√\(?([^)\s*]+)\)?", r"\\sqrt{\1}", text)
    for symbol, latex in _SUPERSCRIPTS.items():
        text = text.replace(symbol, latex)
    text = text.replace("π", "\\pi").replace("×", "\\times").replace("−", "-").replace("°", "^\\circ").replace("÷", "/")
    return text.strip("$ ").strip()


def _fallback_answer_candidates(output_text: str) -> list[str]:
    candidates: list[str] = []
    lines = list(re.finditer(r"\*{0,2}(?:final answer|answer|the final answer is)\*{0,2}\s*:?\*{0,2}\s*(.+)", output_text, re.I))
    if lines:
        candidates.append(lines[-1].group(1).strip())
    bold = re.findall(r"\*\*([^*\n]{1,60})\*\*", output_text[-300:])
    if bold:
        candidates.append(bold[-1].strip())
    return candidates


def numeric_messages(question: str) -> list[ChatMessage]:
    content = (
        "Solve the following math word problem. "
        "Reason step by step, and put the final answer after 'Answer:'.\n\n"
        f"Problem: {question}\n"
    )
    return [{"role": "user", "content": content}]


def boxed_math_messages(question: str) -> list[ChatMessage]:
    content = (
        "Solve the following math problem. "
        "Reason step by step, and put the final answer in \\boxed{}.\n\n"
        f"Problem: {question}\n"
    )
    return [{"role": "user", "content": content}]


def messages_with_solution(messages: list[ChatMessage], solution: str, answer: str) -> list[ChatMessage]:
    content = str(solution).strip()
    if not content:
        content = str(answer)
    return [*messages, {"role": "assistant", "content": content}]


def _snapshot_path(root: str | Path, name: str) -> Path:
    if name not in DATASET_DIRS:
        raise KeyError(f"unknown dataset {name!r}; choose one of {sorted(DATASET_DIRS)}")
    snapshot_root = Path(root) / "snapshots"
    candidates = [snapshot_root / path for path in DATASET_DIRS[name]]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"snapshot not found for {name!r}; tried: {[str(path) for path in candidates]}"
    )


def load_dataset(root: str | Path, name: str, split: str = "test") -> DatasetsDataset:
    path = _snapshot_path(root, name)
    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise KeyError(f"split {split!r} not found for {name}; available: {list(dataset)}")
    return dataset[split]


class MathDataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "test",
        root: str | Path = DEFAULT_ROOT,
        include_raw: bool = False,
        include_answer_message: bool = False,
    ) -> None:
        self.name = name
        self.split = split
        self.include_raw = include_raw
        self.include_answer_message = include_answer_message
        if include_answer_message and split == "test":
            raise ValueError("include_answer_message=True is not allowed for split='test'")
        self.data = self._with_ids(load_dataset(root, name, split))

    def _with_ids(self, data) -> list[dict[str, Any]]:
        rows = []
        for index, row in enumerate(data):
            item = dict(row)
            item["_id"] = f"{self.name}/{self.split}/{index}"
            rows.append(item)
        return rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> C2CExample:
        row = self.data[index]

        if self.name == "gsm8k":
            question = row.get("question", "")
            solution = row.get("answer", "")
            answer = extract_gsm8k_answer(solution)
            answer_type = "numeric"
            messages = numeric_messages(str(question))
            subject = None
        elif self.name in {"math-500", "math-algebra"}:
            question = row.get("problem", row.get("question", ""))
            solution = row.get("solution", "")
            answer = row.get("answer") or extract_boxed_answer(solution) or extract_math_answer(solution)
            answer_type = "math"
            messages = boxed_math_messages(str(question))
            subject = row.get("subject", "algebra" if self.name == "math-algebra" else None)
        else:
            raise KeyError(f"unknown dataset: {self.name}")

        if answer is None:
            raise ValueError(f"could not parse answer for {self.name}/{self.split}/{index}")

        item = {
            "id": row["_id"],
            "dataset": self.name,
            "split": self.split,
            "question": str(question),
            "solution": str(solution),
            "answer": str(answer),
            "answer_type": answer_type,
            "messages": messages,
        }
        if self.include_answer_message:
            item["messages"] = messages_with_solution(messages, item["solution"], item["answer"])
        if subject is not None:
            item["subject"] = str(subject)
        if self.include_raw:
            item["raw"] = row
        return C2CExample(
            id=item["id"],
            dataset=item["dataset"],
            split=item["split"],
            phase1_messages=item["messages"],
            sender_messages=item["messages"],
            receiver_messages=item["messages"],
            target_text=item["answer"],
            prompt_mode="same_prompt_continuation",
            messages=item["messages"],
            metadata=item,
        )

    def select(self, indices: list[int]):
        assert all(0 <= i < len(self.data) for i in indices)
        other = copy.deepcopy(self)
        other.data = [other.data[i] for i in indices]
        return other
