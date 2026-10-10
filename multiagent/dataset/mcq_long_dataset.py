from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset as DatasetsDataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.dataset.mcq_dataset import _answer_index, _choices, answer_letter, grade_output
from multiagent.paths import DATASETS_ROOT


DEFAULT_ROOT = DATASETS_ROOT

DATASET_DIRS = {
    "quality": ["quality"],
    "longhealth": ["longhealth"],
    "race": ["ehovy__race/all"],
    "boolq": ["google__boolq/default"],
}

QUALITY_BOOSTED_SPLIT = "boosted"
QUALITY_BOOSTED_INDICES_PATH = Path(__file__).resolve().parent / "quality_boosted_indices.json"


def messages_from_long_mcq(context: str, question: str, choices: list[str], labels: list[str]) -> list[ChatMessage]:
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    content = (
        "Answer the following multiple choice question using the provided context. "
        "Respond with only the letter of the correct answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Choices:\n"
        f"{choice_lines}\n"
    )
    return [{"role": "user", "content": content}]


def context_messages(context: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            "Use the following patient records as source information for questions that will follow.\n\n"
            f"Context:\n{context}"
        ),
    }]


def quality_context_messages(context: str) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer a question based on the provided context.",
        },
        {"role": "user", "content": context},
    ]


def quality_question_messages(question: str, choices: list[str], labels: list[str]) -> list[ChatMessage]:
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return [{
        "role": "user",
        "content": (
            f"{question}\n\n{choice_lines}\n\n"
            "Please think very briefly and then respond with only the letter "
            "(A, B, C, or D) of the correct option. If you are not sure, still make a guess."
        ),
    }]


def question_messages(question: str, choices: list[str], labels: list[str]) -> list[ChatMessage]:
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return [{
        "role": "user",
        "content": (
            "Answer the following multiple choice question using the patient records above. "
            "Respond with only the letter of the correct answer.\n\n"
            f"Question: {question}\n"
            f"Choices:\n{choice_lines}\n"
        ),
    }]


def document_context_messages(context: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            "Use the following document as source information for a question that will follow.\n\n"
            f"Context:\n{context}"
        ),
    }]


def document_question_messages(question: str, choices: list[str], labels: list[str]) -> list[ChatMessage]:
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return [{
        "role": "user",
        "content": (
            "Answer the following multiple choice question using the document above. "
            "Respond with only the letter of the correct answer.\n\n"
            f"Question: {question}\n"
            f"Choices:\n{choice_lines}\n"
        ),
    }]


def _require(row: dict[str, Any], key: str, sample_id: str) -> Any:
    if key not in row or row[key] is None:
        raise KeyError(f"missing {key!r} in {sample_id}")
    return row[key]


def _long_answer_index(answer: Any, choices: list[str], labels: list[str]) -> int | None:
    if answer is not None:
        answer_text = str(answer).strip()
        for index, choice in enumerate(choices):
            if answer_text == str(choice).strip():
                return index
        for index, label in enumerate(labels):
            if answer_text == label:
                return index
    if isinstance(answer, int):
        if 1 <= answer <= len(choices):
            return answer - 1
    return _answer_index(answer, labels, len(choices))


def _quality_item(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    sample_id = str(row.get("_id", f"quality/{split}/{index}"))
    context = str(_require(row, "article", sample_id))
    question = str(_require(row, "question", sample_id))
    choices, labels = _choices(_require(row, "options", sample_id))
    answer = _require(row, "gold_label", sample_id)
    answer_index = _long_answer_index(answer, choices, labels)
    if answer_index is None:
        raise ValueError(f"could not parse answer {answer!r} in {sample_id}")
    item = {
        "id": sample_id,
        "dataset": "quality",
        "split": split,
        "context": context,
        "question": question,
        "choices": choices,
        "labels": labels,
        "answer_index": answer_index,
        "answer": labels[answer_index],
        "messages": messages_from_long_mcq(context, question, choices, labels),
    }
    if "context_id" in row:
        item["context_id"] = str(row["context_id"])
    if "article_id" in row:
        item["article_id"] = str(row["article_id"])
    return item


def _longhealth_item(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    sample_id = str(row.get("_id", f"longhealth/{split}/{index}"))
    context = str(_require(row, "context", sample_id))
    question = str(_require(row, "question", sample_id))
    choices = [str(_require(row, f"answer_{suffix}", sample_id)) for suffix in ("a", "b", "c", "d", "e")]
    labels = [answer_letter(i) for i in range(len(choices))]
    answer = _require(row, "correct", sample_id)
    answer_index = _long_answer_index(answer, choices, labels)
    if answer_index is None:
        raise ValueError(f"could not parse answer {answer!r} in {sample_id}")
    item = {
        "id": sample_id,
        "dataset": "longhealth",
        "split": split,
        "context": context,
        "question": question,
        "choices": choices,
        "labels": labels,
        "answer_index": answer_index,
        "answer": labels[answer_index],
        "messages": messages_from_long_mcq(context, question, choices, labels),
    }
    if "context_chunks" in row:
        item["context_chunks"] = list(row["context_chunks"])
    if "patient_ids" in row:
        item["patient_ids"] = list(row["patient_ids"])
    if "context_id" in row:
        item["context_id"] = str(row["context_id"])
    if "patient_id" in row:
        item["patient_id"] = str(row["patient_id"])
    return item


def _race_item(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    sample_id = f"race/{split}/{row.get('example_id', index)}"
    context = str(_require(row, "article", sample_id))
    question = str(_require(row, "question", sample_id))
    choices, labels = _choices(_require(row, "options", sample_id))
    answer = _require(row, "answer", sample_id)
    answer_index = _long_answer_index(answer, choices, labels)
    if answer_index is None:
        raise ValueError(f"could not parse answer {answer!r} in {sample_id}")
    context_digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]
    return {
        "id": sample_id,
        "dataset": "race",
        "split": split,
        "context": context,
        "context_id": f"race/context/{context_digest}",
        "question": question,
        "choices": choices,
        "labels": labels,
        "answer_index": answer_index,
        "answer": labels[answer_index],
        "messages": messages_from_long_mcq(context, question, choices, labels),
    }


def _boolq_item(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    sample_id = f"boolq/{split}/{index}"
    context = str(_require(row, "passage", sample_id))
    question = str(_require(row, "question", sample_id))
    choices = ["No", "Yes"]
    labels = ["A", "B"]
    answer_index = 1 if bool(_require(row, "answer", sample_id)) else 0
    context_digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]
    return {
        "id": sample_id,
        "dataset": "boolq",
        "split": split,
        "context": context,
        "context_id": f"boolq/context/{context_digest}",
        "question": question,
        "choices": choices,
        "labels": labels,
        "answer_index": answer_index,
        "answer": labels[answer_index],
        "messages": messages_from_long_mcq(context, question, choices, labels),
    }


def _group_longhealth_patients(data, patients_per_context: int = 5) -> list[dict[str, Any]]:
    patients = [dict(row) for row in data]
    if not patients:
        return []
    if patients_per_context <= 0:
        raise ValueError("patients_per_context must be positive")
    groups = []
    for group_start in range(0, len(patients), patients_per_context):
        group = patients[group_start : group_start + patients_per_context]
        group_index = group_start // patients_per_context
        patient_ids = [str(patient.get("id", group_start + offset)) for offset, patient in enumerate(group)]
        context_chunks = [str(_require(patient, "context", patient_ids[offset])) for offset, patient in enumerate(group)]
        questions = []
        for patient, patient_id in zip(group, patient_ids):
            for question in _require(patient, "questions", patient_id):
                questions.append({**dict(question), "patient_id": patient_id})
        expected_questions = 20 * len(group)
        if len(questions) != expected_questions:
            raise ValueError(
                f"LongHealth context {group_index} has {len(questions)} questions; "
                f"expected {expected_questions} for {len(group)} patients"
            )
        context_prefix = (
            "longhealth"
            if patients_per_context == 5
            else f"longhealth/ppc={patients_per_context}"
        )
        groups.append({
            "id": f"context_{group_index + 1:02d}",
            "context_id": f"{context_prefix}/context/{group_index + 1:02d}",
            "patient_ids": patient_ids,
            "context_chunks": context_chunks,
            "context": "\n\n".join(context_chunks),
            "questions": questions,
        })
    return groups


def _group_quality_articles(data) -> list[dict[str, Any]]:
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, source_row in enumerate(data):
        row = dict(source_row)
        article = str(_require(row, "article", f"quality/dev/{row_index}")).strip()
        article_id = str(row.get("article_id", row.get("id", row_index)))
        key = (article_id, article)
        questions = list(_require(row, "questions", f"quality/dev/{row_index}"))
        if key in combined:
            combined[key]["questions"].extend(questions)
            continue
        combined[key] = {
            **row,
            "article": article,
            "article_id": article_id,
            "context_id": f"quality/context/{article_id}",
            "questions": questions,
        }
    return list(combined.values())


def _snapshot_path(root: str | Path, name: str) -> Path:
    if name not in DATASET_DIRS:
        raise KeyError(f"unknown long MCQ dataset {name!r}; choose one of {sorted(DATASET_DIRS)}")
    candidates = [Path(root) / "snapshots" / relative for relative in DATASET_DIRS[name]]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "snapshot not found for "
        f"{name!r}; tried: {', '.join(str(path) for path in candidates)}. "
        f"Save/download it under one of those names."
    )


def load_dataset(root: str | Path, name: str, split: str = "test") -> list[dict[str, Any]] | DatasetsDataset:
    path = _snapshot_path(root, name)
    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise KeyError(f"split {split!r} not found for {name}; available: {list(dataset)}")
    return dataset[split]


class MCQLongDataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "test",
        root: str | Path = DEFAULT_ROOT,
        include_raw: bool = False,
        include_answer_message: bool = False,
        patients_per_context: int = 5,
    ) -> None:
        self.name = name
        self.split = split
        self.include_raw = include_raw
        self.include_answer_message = include_answer_message
        self.patients_per_context = patients_per_context
        self.preserve_order = name in {"quality", "longhealth", "race", "boolq"}
        if include_answer_message and split == "test":
            raise ValueError("include_answer_message=True is not allowed for split='test'")
        source_split = split
        boosted_indices = None
        if name == "quality" and split == QUALITY_BOOSTED_SPLIT:
            source_split = "dev"
            boosted_indices = json.loads(QUALITY_BOOSTED_INDICES_PATH.read_text())["indices"]
        self.data = self._flatten(load_dataset(root, name, source_split))
        if boosted_indices is not None:
            self.data = [self.data[i] for i in boosted_indices]

    def _flatten(self, data) -> list[dict[str, Any]]:
        if self.name in {"race", "boolq"}:
            return [dict(row) for row in data]
        if self.name == "quality":
            data = _group_quality_articles(data)
        elif self.name == "longhealth":
            data = _group_longhealth_patients(data, self.patients_per_context)
        rows = []
        for row_index, row in enumerate(data):
            parent = dict(row)
            questions = _require(parent, "questions", f"{self.name}/{self.split}/{row_index}")
            if not isinstance(questions, list):
                raise TypeError(f"expected questions list in {self.name}/{self.split}/{row_index}")
            for question_index, question_row in enumerate(questions):
                question = dict(question_row)
                if self.name == "quality":
                    item = {
                        "article": _require(parent, "article", f"quality/{self.split}/{row_index}"),
                        "article_id": _require(parent, "article_id", f"quality/{self.split}/{row_index}"),
                        "context_id": _require(parent, "context_id", f"quality/{self.split}/{row_index}"),
                        **question,
                    }
                    item["_id"] = f"quality/{self.split}/{parent.get('id', row_index)}/{question_index}"
                elif self.name == "longhealth":
                    item = {
                        "context": _require(parent, "context", f"longhealth/{self.split}/{row_index}"),
                        "context_chunks": _require(parent, "context_chunks", f"longhealth/{self.split}/{row_index}"),
                        "context_id": _require(parent, "context_id", f"longhealth/{self.split}/{row_index}"),
                        "patient_ids": _require(parent, "patient_ids", f"longhealth/{self.split}/{row_index}"),
                        **question,
                    }
                    item["_id"] = f"longhealth/{self.split}/{parent.get('id', row_index)}/{question_index}"
                else:
                    raise KeyError(f"unknown long MCQ dataset: {self.name}")
                rows.append(item)
        return rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> C2CExample:
        row = self.data[index]
        if self.name == "quality":
            item = _quality_item(row, self.split, index)
        elif self.name == "longhealth":
            item = _longhealth_item(row, self.split, index)
        elif self.name == "race":
            item = _race_item(row, self.split, index)
        elif self.name == "boolq":
            item = _boolq_item(row, self.split, index)
        else:
            raise KeyError(f"unknown long MCQ dataset: {self.name}")
        if self.include_answer_message:
            answer_index = item["answer_index"]
            item["messages"] = [
                *item["messages"],
                {"role": "assistant", "content": f"The correct answer is {item['labels'][answer_index]}. {item['choices'][answer_index]}"},
            ]
        if self.include_raw:
            item["raw"] = row
        if self.name == "quality":
            sender_messages = quality_context_messages(item["context"])
            receiver_messages = quality_question_messages(item["question"], item["choices"], item["labels"])
            item["messages"] = [*sender_messages, *receiver_messages]
            prompt_mode = "split_prompt"
        elif self.name == "longhealth":
            sender_messages = context_messages(item["context"])
            receiver_messages = question_messages(item["question"], item["choices"], item["labels"])
            prompt_mode = "split_prompt"
        elif self.name in {"race", "boolq"}:
            sender_messages = document_context_messages(item["context"])
            receiver_messages = document_question_messages(item["question"], item["choices"], item["labels"])
            prompt_mode = "split_prompt"
        else:
            sender_messages = item["messages"]
            receiver_messages = item["messages"]
            prompt_mode = "same_prompt_continuation"
        return C2CExample(
            id=item["id"],
            dataset=item["dataset"],
            split=item["split"],
            phase1_messages=item["messages"],
            sender_messages=sender_messages,
            receiver_messages=receiver_messages,
            target_text=item["answer"],
            prompt_mode=prompt_mode,
            messages=item["messages"],
            metadata=item,
        )

    def select(self, indices: list[int]):
        assert all(0 <= i < len(self.data) for i in indices)
        other = copy.deepcopy(self)
        other.data = [other.data[i] for i in indices]
        return other
