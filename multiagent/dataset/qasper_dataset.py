from __future__ import annotations

import copy
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset as DatasetsDataset
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.paths import DATASETS_ROOT


DEFAULT_ROOT = DATASETS_ROOT
DATASET_DIR = "allenai__qasper"
QASPER_TASKS = {"qasper"}
QASPER_ANSWER_INSTRUCTION = "Answer as briefly as possible. Give only the answer, no explanation."
QASPER_NO_UNANSWERABLE_OPTION = "nox"
QASPER_LENGTH_OPTIONS = {"length_model", "min_tokens", "max_tokens"}


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
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
    return output.strip()


def answer_reference(answer_info: dict[str, Any]) -> dict[str, Any]:
    answer = answer_info["answer"]
    if answer.get("unanswerable"):
        return {"answer": "Unanswerable", "type": "none", "evidence": []}
    if answer.get("extractive_spans"):
        return {"answer": ", ".join(map(str, answer["extractive_spans"])), "type": "extractive", "evidence": answer.get("evidence", [])}
    if answer.get("free_form_answer"):
        return {"answer": str(answer["free_form_answer"]), "type": "abstractive", "evidence": answer.get("evidence", [])}
    if answer.get("yes_no") is True:
        return {"answer": "Yes", "type": "boolean", "evidence": answer.get("evidence", [])}
    if answer.get("yes_no") is False:
        return {"answer": "No", "type": "boolean", "evidence": answer.get("evidence", [])}
    raise ValueError("Qasper answer annotation does not contain an answer")


def phase2_loss_weight(answer_type: str) -> float:
    if answer_type in {"extractive", "abstractive", "free_text"}:
        return 1.0
    if answer_type in {"boolean", "none", "unanswerable"}:
        return 0.01
    raise ValueError(f"unknown Qasper answer type {answer_type!r}")


def is_unanswerable_type(answer_type: Any) -> bool:
    return str(answer_type) in {"none", "unanswerable"}


def validate_options(options: dict[str, str]) -> None:
    unsupported = set(options) - {QASPER_NO_UNANSWERABLE_OPTION} - QASPER_LENGTH_OPTIONS
    if unsupported:
        raise ValueError(f"unsupported Qasper options: {options}")
    if options.get(QASPER_NO_UNANSWERABLE_OPTION, ""):
        raise ValueError(f"{QASPER_NO_UNANSWERABLE_OPTION!r} is a flag and should be used without a value")


def grade_output(output_text: str, sample: dict[str, Any]) -> dict[str, Any]:
    references = sample.get("references", [])
    if not references:
        return {"grade": "fail", "answer_f1": 0.0}
    prediction = extract_answer(output_text)
    scored = [
        (token_f1_score(prediction, str(reference["answer"])), str(reference.get("type", "unknown")))
        for reference in references
    ]
    answer_f1, answer_type = max(scored, key=lambda item: item[0])
    return {
        "grade": "correct" if answer_f1 >= 1.0 else "incorrect",
        "answer_f1": answer_f1,
        "answer_type": answer_type,
    }


def format_document(row: dict[str, Any]) -> str:
    parts = []
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if title:
        parts.append(f"Title:\n{title}")
    if abstract:
        parts.append(f"Abstract:\n{abstract}")
    for section in row.get("full_text") or []:
        name = str(section.get("section_name") or "Section").strip()
        paragraphs = [str(paragraph).strip() for paragraph in section.get("paragraphs") or [] if str(paragraph).strip()]
        if paragraphs:
            parts.append(f"{name}:\n" + "\n\n".join(paragraphs))
    return "\n\n".join(parts).strip()


def qasper_messages(question: str, context: str) -> list[ChatMessage]:
    return [{
        "role": "user",
        "content": (
            "Answer the question using the provided research paper.\n\n"
            f"Research paper:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"{QASPER_ANSWER_INSTRUCTION}"
        ),
    }]


def load_dataset(root: str | Path, split: str = "validation") -> DatasetsDataset:
    path = Path(root) / "snapshots" / DATASET_DIR
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}. Download it with: python multiagent/dataset/download_hf.py qasper")
    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise KeyError(f"split {split!r} not found for qasper; available: {list(dataset)}")
    return dataset[split]


class QasperDataset(Dataset):
    def __init__(
        self,
        split: str = "validation",
        root: str | Path = DEFAULT_ROOT,
        include_raw: bool = False,
        exclude_unanswerable: bool = False,
        length_model: str | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.split = split
        self.include_raw = include_raw
        self.exclude_unanswerable = exclude_unanswerable
        self.length_model = length_model
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.data = self._flatten(load_dataset(root, split))
        if min_tokens is not None or max_tokens is not None:
            if not length_model:
                raise ValueError("length_model is required with min_tokens or max_tokens")
            if min_tokens is not None and min_tokens < 1:
                raise ValueError("min_tokens must be positive")
            if max_tokens is not None and min_tokens is not None and max_tokens < min_tokens:
                raise ValueError("max_tokens must be greater than or equal to min_tokens")
            tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(length_model))
            length_cache: dict[str, int] = {}
            filtered = []
            for row in self.data:
                paper_id = str(row["paper_id"])
                token_length = length_cache.get(paper_id)
                if token_length is None:
                    token_length = len(tokenizer(row["context"], add_special_tokens=False)["input_ids"])
                    length_cache[paper_id] = token_length
                if min_tokens is not None and token_length < min_tokens:
                    continue
                if max_tokens is not None and token_length > max_tokens:
                    continue
                item = dict(row)
                item["selection_token_length"] = token_length
                item["selection_length_model"] = length_model
                filtered.append(item)
            self.data = filtered

    def _flatten(self, data) -> list[dict[str, Any]]:
        rows = []
        for paper_index, paper_row in enumerate(data):
            paper = dict(paper_row)
            context = format_document(paper)
            paper_id = str(paper.get("id", paper_index))
            for question_index, qa_row in enumerate(paper.get("qas") or []):
                references = [answer_reference(answer) for answer in qa_row.get("answers") or []]
                if not references:
                    continue
                if self.exclude_unanswerable and is_unanswerable_type(references[0]["type"]):
                    continue
                question = str(qa_row.get("question", ""))
                item = {
                    "id": f"qasper/{self.split}/{paper_id}/{qa_row.get('question_id', question_index)}",
                    "dataset": "qasper",
                    "split": self.split,
                    "paper_id": paper_id,
                    "context_id": f"qasper/paper/{self.split}/{paper_id}",
                    "question_id": str(qa_row.get("question_id", "")),
                    "title": paper.get("title"),
                    "context": context,
                    "question": question,
                    "references": references,
                    "answers": [reference["answer"] for reference in references],
                    "answer": references[0]["answer"],
                    "answer_type": references[0]["type"],
                    "phase2_loss_weight": phase2_loss_weight(references[0]["type"]),
                    "messages": qasper_messages(question, context),
                }
                if self.include_raw:
                    item["raw"] = {"paper": paper, "qa": qa_row}
                rows.append(item)
        return rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> C2CExample:
        item = self.data[index]
        context = item["context"]
        question = item["question"]
        phase1_messages = [{
            "role": "user",
            "content": (
                "Use the following research paper as source information for a later question.\n\n"
                f"Research paper:\n{context}\n\n"
                "Answer the question using the research paper above.\n\n"
                f"Question:\n{question}\n\n"
                f"{QASPER_ANSWER_INSTRUCTION}"
            ),
        }]
        sender_messages = [{
            "role": "user",
            "content": (
                "Use the following research paper as source information for a later question.\n\n"
                f"Research paper:\n{context}"
            ),
        }]
        receiver_messages = [{
            "role": "user",
            "content": (
                "Answer the question using the research paper above.\n\n"
                f"Question:\n{question}\n\n"
                f"{QASPER_ANSWER_INSTRUCTION}"
            ),
        }]
        return C2CExample(
            id=item["id"],
            dataset=item["dataset"],
            split=item["split"],
            phase1_messages=phase1_messages,
            sender_messages=sender_messages,
            receiver_messages=receiver_messages,
            target_text=item["answer"],
            prompt_mode="split_prompt",
            messages=item["messages"],
            metadata=item,
        )

    def select(self, indices: list[int]):
        assert all(0 <= i < len(self.data) for i in indices)
        other = copy.deepcopy(self)
        other.data = [other.data[i] for i in indices]
        return other
