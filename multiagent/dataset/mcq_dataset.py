from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import copy

from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset as DatasetsDataset

from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.paths import DATASETS_ROOT


DEFAULT_ROOT = DATASETS_ROOT

MMMLU_LANGUAGES = ("AR-XY", "BN-BD", "DE-DE", "ES-LA", "FR-FR", "HI-IN", "ID-ID", "IT-IT", "JA-JP", "KO-KR",
                   "PT-BR", "SW-KE", "YO-NG", "ZH-CN")

DATASET_DIRS = {
    "mmlu-redux": "edinburgh-dawg__mmlu-redux-2.0",
    "openbookqa": "allenai__openbookqa/main",
    "arc-c": "allenai__ai2_arc/ARC-Challenge",
    "arc-e": "allenai__ai2_arc/ARC-Easy",
    "medqa": "GBaker__MedQA-USMLE-4-options/default",
    "medmcqa": "openlifescienceai__medmcqa/default",
    "ceval": "ceval__ceval-exam",
    "cmmlu": "haonan-li__cmmlu",
    "hellaswag": "Rowan__hellaswag/default",
    "winogrande": "allenai__winogrande/winogrande_xl",
    "mmlu": "cais__mmlu/all",
    **{f"mmmlu-{code.split('-')[0].lower()}": "openai__MMMLU" for code in MMMLU_LANGUAGES},
}

CHINESE_MCQ = {"ceval", "cmmlu"}
CSV_MCQ = CHINESE_MCQ | {name for name in DATASET_DIRS if name.startswith("mmmlu-")}


def answer_letter(index: int) -> str:
    return chr(ord("A") + index)


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_question(row: dict[str, Any]) -> tuple[str, list[str]]:
    ctx = f"{row['activity_label']}: {row['ctx_a']} {str(row['ctx_b']).capitalize()}"
    question = "Which option is the most plausible continuation of the following text?\n\n" + _hellaswag_clean(ctx)
    return question, [_hellaswag_clean(str(e)) for e in row["endings"]]


def winogrande_question(row: dict[str, Any]) -> tuple[str, list[str]]:
    question = "Which option correctly fills in the blank (_) in the following sentence?\n\n" + str(row["sentence"])
    return question, [str(row["option1"]), str(row["option2"])]


def messages_from_mcq(question: str, choices: list[str], labels: list[str]) -> list[ChatMessage]:
    choice_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    content = (
        "Answer the following multiple choice question. "
        "Respond with only the letter of the correct answer.\n\n"
        f"Question: {question}\n"
        f"Choices:\n"
        f"{choice_lines}\n"
    )
    return [{"role": "user", "content": content}]


def messages_with_answer(messages: list[ChatMessage], answer: str, answer_text: str) -> list[ChatMessage]:
    content = f"The correct answer is {answer}. {answer_text}"
    return [*messages, {"role": "assistant", "content": content}]


def parse_choice(output_text: str, labels: list[str]) -> str | None:
    labels = [label.upper() for label in labels]
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    text = output_text.strip()

    if text.upper() in labels:
        return text.upper()

    normalized = text.upper()
    for _ in range(3):
        normalized = re.sub(r"\\(?:BOXED|TEXT|MATHRM)\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = normalized.replace("$", " ").replace("`", " ").replace("*", " ")

    cues = [
        rf"\b(?:ANSWER|ANS)\s*(?:IS|:|=)+\s*[:=]?\s*[\(\[\{{\s]*({label_pattern})\b",
        rf"\b(?:CHOICE|OPTION)\s*(?:IS|:|=)+\s*[:=]?\s*[\(\[\{{\s]*({label_pattern})\b",
        rf"\b(?:CHOOSE|CHOSE|SELECT|SELECTED|PICK|PICKED)\s*[\(\[\{{\s]*({label_pattern})\b",
    ]
    hits: list[tuple[int, str]] = []
    for pattern in cues:
        hits.extend((m.start(1), m.group(1)) for m in re.finditer(pattern, normalized))
    if hits:
        return max(hits, key=lambda item: item[0])[1]

    for label in labels:
        if text.upper().startswith(f"{label}.") or text.upper().startswith(f"{label})"):
            return label
    leading = re.match(rf"^\s*[\(\[\{{\s]*({label_pattern})\b", normalized)
    if leading:
        return leading.group(1)

    per_line = {m.group(1) for m in re.finditer(rf"(?m)^\s*[\(\[\{{\s]*({label_pattern})[\.\)\:]", normalized)}
    if len(per_line) == 1:
        return per_line.pop()
    return None


def grade_output(output_text: str, sample: dict[str, Any]) -> str:
    labels = [str(label).strip().upper() for label in sample["labels"]]
    answer = str(sample["answer"]).strip().upper()
    parsed = parse_choice(output_text, labels)
    if parsed is None:
        return "fail"
    return "correct" if parsed == answer else "incorrect"


def _choices(raw_choices: Any) -> tuple[list[str], list[str]]:
    if isinstance(raw_choices, dict):
        if "text" in raw_choices:
            texts = [str(item) for item in raw_choices.get("text", [])]
            labels = [str(item) for item in raw_choices.get("label", [])]
        else:
            items = sorted(raw_choices.items(), key=lambda item: str(item[0]))
            labels = [str(label) for label, _ in items]
            texts = [str(text) for _, text in items]
        if not labels:
            labels = [answer_letter(i) for i in range(len(texts))]
        return texts, labels

    texts = []
    labels = []
    for index, item in enumerate(raw_choices or []):
        if isinstance(item, dict):
            texts.append(str(item.get("text", "")))
            labels.append(str(item.get("label", answer_letter(index))))
        else:
            texts.append(str(item))
            labels.append(answer_letter(index))
    return texts, labels


def _answer_index(answer: Any, labels: list[str], num_choices: int) -> int | None:
    if isinstance(answer, int):
        return answer if 0 <= answer < num_choices else None

    text = str(answer).strip()
    if text in labels:
        return labels.index(text)
    if len(text) == 1 and "A" <= text.upper() <= "Z":
        index = ord(text.upper()) - ord("A")
        return index if 0 <= index < num_choices else None
    if text.isdigit():
        value = int(text)
        if 0 <= value < num_choices:
            return value
        if 1 <= value <= num_choices:
            return value - 1
    return None


def _snapshot_path(root: str | Path, name: str) -> Path:
    if name not in DATASET_DIRS:
        raise KeyError(f"unknown dataset {name!r}; choose one of {sorted(DATASET_DIRS)}")
    return Path(root) / "snapshots" / DATASET_DIRS[name]


def mmlu_subjects(root: str | Path, subjects: list[str] | None = None) -> list[str]:
    path = _snapshot_path(root, "mmlu-redux")
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot not found: {path}. Download it with: python scripts/download/download_eval_datasets.py"
        )
    available = sorted(child.name for child in path.iterdir() if (child / "dataset_dict.json").exists())
    if subjects is None:
        return available
    missing = sorted(set(subjects) - set(available))
    if missing:
        raise FileNotFoundError(f"missing MMLU-Redux subjects: {missing}")
    return subjects


def load_dataset(
    root: str | Path,
    name: str,
    split: str = "test",
    mmlu_subjects_arg: list[str] | None = None,
) -> list[dict[str, Any]] | DatasetsDataset:
    path = _snapshot_path(root, name)
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot not found: {path}. Download it with: python scripts/download/download_eval_datasets.py"
        )

    if name == "mmlu-redux":
        rows = []
        for subject in mmlu_subjects(root, mmlu_subjects_arg):
            dataset = load_from_disk(str(path / subject))
            if split not in dataset:
                raise KeyError(f"split {split!r} not found for {name}/{subject}; available: {list(dataset)}")
            for row_index, row in enumerate(dataset[split]):
                item = dict(row)
                item["_subject"] = subject
                item["_row"] = row_index
                rows.append(item)
        return rows

    if name in CSV_MCQ:
        return _load_chinese_mcq(path, name, split)

    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise KeyError(f"split {split!r} not found for {name}; available: {list(dataset)}")
    return dataset[split]


def _load_chinese_mcq(path: Path, name: str, split: str) -> list[dict[str, Any]]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    if name == "ceval":
        files = sorted(path.glob(f"*/{split}-*.parquet"))
        if not files:
            raise KeyError(f"split {split!r} not found for ceval under {path}")
        frames = [(f.parent.name, pd.read_parquet(f)) for f in files]
    elif name.startswith("mmmlu-"):
        code = next(c for c in MMMLU_LANGUAGES if c.split("-")[0].lower() == name[len("mmmlu-"):])
        csv = path / split / f"mmlu_{code}.csv"
        if not csv.is_file():
            raise KeyError(f"split {split!r} not found for {name} under {path}")
        frame = pd.read_csv(csv)
        frames = [(subject, group.reset_index(drop=True)) for subject, group in frame.groupby("Subject", sort=True)]
    else:
        half = None
        if split in ("test_a", "test_b"):
            half, split = (0 if split == "test_a" else 1), "test"
        split_dir = path / split
        if not split_dir.is_dir():
            raise KeyError(f"split {split!r} not found for cmmlu under {path}")
        files = sorted(split_dir.glob("*.csv"))
        if half is not None:
            files = [f for i, f in enumerate(files) if i % 2 == half]
        frames = [(f.stem, pd.read_csv(f)) for f in files]
    for subject, frame in frames:
        for row_index, row in enumerate(frame.to_dict("records")):
            question = str(row.get("Question", row.get("question", "")))
            choices = [str(row[label]) for label in ("A", "B", "C", "D") if label in row]
            rows.append({
                "question": question,
                "choices": choices,
                "answer": str(row.get("Answer", row.get("answer", ""))).strip(),
                "_subject": subject,
                "_row": row_index,
            })
    return rows


class MCQDataset(Dataset):
    def __init__(
        self,
        name: str,
        split: str = "test",
        root: str | Path = DEFAULT_ROOT,
        mmlu_subjects_arg: list[str] | None = None,
        include_raw: bool = False,
        include_answer_message: bool = False,
    ) -> None:
        self.name = name
        self.split = split
        self.include_raw = include_raw
        self.include_answer_message = include_answer_message
        if include_answer_message and split == "test":
            raise ValueError("include_answer_message=True is not allowed for split='test'")
        self.data = self._with_ids(load_dataset(root, name, split, mmlu_subjects_arg))

    def _with_ids(self, data) -> list[dict[str, Any]]:
        rows = []
        for index, row in enumerate(data):
            item = dict(row)
            if self.name == "mmlu-redux" or self.name in CSV_MCQ:
                row_id = f"{item.get('_subject')}/{item.get('_row', index)}"
            elif self.name in {"openbookqa", "arc-c", "arc-e"}:
                row_id = item.get("id", index)
            elif self.name == "mmlu":
                row_id = f"{item.get('subject')}/{index}"
            elif self.name == "medmcqa":
                row_id = item.get("id", index)
            else:
                row_id = index
            item["_id"] = f"{self.name}/{self.split}/{row_id}"
            rows.append(item)
        return rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> C2CExample:
        row = self.data[index]
        choices, labels = _choices(row.get("choices", row.get("options")))

        if self.name == "mmlu-redux":
            answer = row.get("answer", row.get("correct_answer"))
            question = row.get("question", "")
            subject = row.get("_subject")
        elif self.name == "openbookqa":
            answer = row.get("answerKey")
            question = row.get("question_stem", "")
            subject = None
        elif self.name in {"arc-c", "arc-e"}:
            answer = row.get("answerKey")
            question = row.get("question", "")
            subject = None
            if labels and all(label.isdigit() for label in labels):
                answer = answer_letter(labels.index(str(answer).strip()))
                labels = [answer_letter(i) for i in range(len(labels))]
        elif self.name == "medqa":
            answer = row.get("answer_idx", row.get("answer"))
            question = row.get("question", "")
            subject = row.get("meta_info")
        elif self.name == "medmcqa":
            choices = [str(row[key]) for key in ("opa", "opb", "opc", "opd")]
            labels = [answer_letter(i) for i in range(4)]
            answer = int(row["cop"])
            question = row.get("question", "")
            subject = row.get("subject_name")
        elif self.name in CSV_MCQ:
            answer = row.get("answer")
            question = row.get("question", "")
            subject = row.get("_subject")
        elif self.name == "hellaswag":
            question, choices = hellaswag_question(row)
            labels = [answer_letter(i) for i in range(len(choices))]
            answer = int(row["label"])
            subject = row.get("activity_label")
        elif self.name == "winogrande":
            question, choices = winogrande_question(row)
            labels = ["A", "B"]
            answer = int(row["answer"]) - 1
            subject = None
        elif self.name == "mmlu":
            answer = int(row["answer"])
            question = row.get("question", "")
            subject = row.get("subject")
        else:
            raise KeyError(f"unknown dataset: {self.name}")

        answer_index = _answer_index(answer, labels, len(choices))
        if answer_index is None:
            raise ValueError(f"could not parse answer {answer!r} for {self.name}/{self.split}/{index}")

        messages = messages_from_mcq(str(question), choices, labels)
        if self.include_answer_message:
            messages = messages_with_answer(messages, labels[answer_index], choices[answer_index])

        item = {
            "id": row["_id"],
            "dataset": self.name,
            "split": self.split,
            "question": str(question),
            "choices": choices,
            "labels": labels,
            "answer_index": answer_index,
            "answer": labels[answer_index],
            "messages": messages,
        }
        if subject is not None:
            item["subject"] = subject
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
