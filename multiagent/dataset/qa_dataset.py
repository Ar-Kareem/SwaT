from __future__ import annotations

import copy
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset as DatasetsDataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.paths import DATASETS_ROOT


DEFAULT_ROOT = DATASETS_ROOT

DATASET_DIRS = {
    "2wikimqa": "voidful__2WikiMultihopQA/default",
    "hotpotqa": "hotpotqa__hotpot_qa/distractor",
    "musique": "voidful__MuSiQue/default",
    "squad": "rajpurkar__squad/plain_text",
    "narrativeqa": "deepmind__narrativeqa/default",
}

QA_TASKS = set(DATASET_DIRS)


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(output: str) -> str:
    output = str(output).strip()
    for marker in ("<|eot_id|>", "<|end_of_text|>"):
        output = output.split(marker)[0]
    if "<|start_header_id|>assistant<|end_header_id|>" in output:
        output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    output = output.strip()
    output = re.sub(r"^(answer|the answer is)\s*:\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"^the answer is\s+", "", output, flags=re.IGNORECASE)
    return output.strip()


def best_subspan_em(prediction: str, answers: list[str]) -> bool:
    prediction = normalize_answer(prediction)
    for answer in answers:
        answer = normalize_answer(answer)
        if answer and answer in prediction:
            return True
    return False


def squad_exact_match(prediction: str, answers: list[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(answer) for answer in answers))


def squad_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if not same:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def grade_output(output_text: str, sample: dict[str, Any]) -> str | dict[str, Any]:
    answers = [str(answer) for answer in sample.get("answers", [])]
    if not answers:
        return "fail"
    prediction = extract_answer(output_text)
    return {
        "grade": "correct" if best_subspan_em(prediction, answers) else "incorrect",
        "exact_match": squad_exact_match(prediction, answers),
        "f1": squad_f1(prediction, answers),
    }


INSTRUCTIONS = {
    "default": "Answer the question using the provided context. Respond with a concise answer.",
    "terse": "Answer the question based on the given passages. \nOnly give me the answer and do not output any other words."
}

def qa_instruction(prompt_style: str = "default") -> str:
    if prompt_style not in INSTRUCTIONS:
        raise ValueError(f"unknown QA prompt style {prompt_style!r}; choose from {sorted(INSTRUCTIONS)}")
    return INSTRUCTIONS[prompt_style]


def qa_messages(question: str, context: str, prompt_style: str = "default") -> list[ChatMessage]:
    instruction = qa_instruction(prompt_style)
    answer_cue = "Answer:" if prompt_style == "terse" else ""
    content = (
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n{answer_cue}"
    )
    return [{"role": "user", "content": content}]


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _answers(value: Any) -> list[str]:
    value = _maybe_json(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    value = str(value)
    return [value] if value.strip() else []


def _format_doc(title: Any, text: Any) -> str:
    title = "" if title is None else str(title).strip()
    text = "" if text is None else str(text).strip()
    return f"{title}\n{text}" if title else text


def _normalize_context(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "squad":
        return str(row.get("context", ""))

    if dataset == "narrativeqa":
        return str(row["document"]["summary"]["text"]).strip()

    if dataset == "2wikimqa":
        context = _maybe_json(row.get("context", []))
        return "\n\n".join(_format_doc(title, " ".join(map(str, sentences))) for title, sentences in context)

    if dataset == "hotpotqa":
        context = row.get("context", {})
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return "\n\n".join(_format_doc(title, " ".join(map(str, sent))) for title, sent in zip(titles, sentences))

    if dataset == "musique":
        paragraphs = _maybe_json(row.get("paragraphs", []))
        return "\n\n".join(_format_doc(item.get("title"), item.get("paragraph_text", "")) for item in paragraphs)

    raise KeyError(f"unknown dataset: {dataset}")


def load_dataset(root: str | Path, name: str, split: str = "train") -> list | DatasetsDataset:
    if name not in DATASET_DIRS:
        raise KeyError(f"unknown dataset {name!r}; choose one of {sorted(DATASET_DIRS)}")
    path = Path(root) / "snapshots" / DATASET_DIRS[name]
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}. Download it with: python multiagent/dataset/download_hf.py {name}")
    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise KeyError(f"split {split!r} not found for {name}; available: {list(dataset)}")
    return dataset[split]


class QADataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "train",
        root: str | Path = DEFAULT_ROOT,
        include_raw: bool = False,
        prompt_style: str = "default",
    ) -> None:
        self.name = name
        self.split = split
        self.include_raw = include_raw
        self.prompt_style = prompt_style
        qa_instruction(prompt_style)
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
        question = row.get("question", "")
        if isinstance(question, dict):
            question = question.get("text", "")
        question = str(question)
        raw_answers = row.get("answers", {}) if self.name == "squad" else row.get("answer")
        if self.name == "squad" and isinstance(raw_answers, dict):
            raw_answers = raw_answers.get("text", [])
        if self.name == "narrativeqa":
            raw_answers = [answer["text"] for answer in row.get("answers", [])]
        answers = _answers(raw_answers)
        context = _normalize_context(self.name, row)
        item = {
            "id": row["_id"],
            "dataset": self.name,
            "split": self.split,
            "question": question,
            "answers": answers,
            "answer": answers,
            "context": context,
            "messages": qa_messages(question, context, self.prompt_style),
        }
        if self.include_raw:
            item["raw"] = row
        target_text = answers[0] if answers else None
        instruction = qa_instruction(self.prompt_style).replace("the provided context", "the context above")
        answer_cue = "\nAnswer:" if self.prompt_style == "terse" else ""
        phase1_messages = [{
            "role": "user",
            "content": (
                "Use the following context as source information for a later question.\n\n"
                f"Context:\n{context}\n\n"
                f"{instruction}\n\n"
                f"Question:\n{question}{answer_cue}"
            ),
        }]
        sender_messages = [{
            "role": "user",
            "content": (
                "Use the following context as source information for a later question.\n\n"
                f"Context:\n{context}"
            ),
        }]
        receiver_messages = [{
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Question:\n{question}{answer_cue}"
            ),
        }]
        return C2CExample(
            id=item["id"],
            dataset=item["dataset"],
            split=item["split"],
            phase1_messages=phase1_messages,
            sender_messages=sender_messages,
            receiver_messages=receiver_messages,
            target_text=target_text,
            prompt_mode="split_prompt",
            messages=item["messages"],
            metadata=item,
        )

    def select(self, indices: list[int]):
        assert all(0 <= i < len(self.data) for i in indices)
        other = copy.deepcopy(self)
        other.data = [other.data[i] for i in indices]
        return other
