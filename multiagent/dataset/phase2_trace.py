from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample
from multiagent.dataset.math_dataset import DATASET_DIRS as MATH_DATASET_DIRS
from multiagent.dataset.math_dataset import boxed_math_messages, numeric_messages
from multiagent.dataset.mcq_dataset import DATASET_DIRS as MCQ_DATASET_DIRS
from multiagent.dataset.mcq_dataset import messages_from_mcq


MATH_TASKS = set(MATH_DATASET_DIRS)
MCQ_TASKS = set(MCQ_DATASET_DIRS)


class Phase2TraceJsonlDataset(Dataset):

    def __init__(self, path: str | Path):
        self.path = Path(path)
        rows: list[C2CExample] = []
        with open(self.path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(self._load_row(json.loads(line), line_number))
        self.rows = rows

    def _load_row(self, row: dict[str, Any], line_number: int) -> C2CExample:
        for key in ("id", "dataset", "split", "question", "trace"):
            if key not in row:
                raise ValueError(f"missing phase-2 trace field {key!r} in {self.path}:{line_number}")

        task = str(row["dataset"])
        question = str(row["question"])
        if task == "gsm8k":
            messages = numeric_messages(question)
        elif task in MATH_TASKS:
            messages = boxed_math_messages(question)
        elif task in MCQ_TASKS:
            for key in ("choices", "labels"):
                if key not in row:
                    raise ValueError(f"missing phase-2 trace field {key!r} in {self.path}:{line_number}")
            messages = messages_from_mcq(question, [str(item) for item in row["choices"]], [str(item) for item in row["labels"]])
        else:
            raise KeyError(f"unsupported phase-2 trace dataset {task!r} in {self.path}:{line_number}")

        return C2CExample(
            id=str(row["id"]),
            dataset=task,
            split=str(row["split"]),
            phase1_messages=messages,
            sender_messages=messages,
            receiver_messages=messages,
            target_text=str(row["trace"]),
            prompt_mode="same_prompt_continuation",
            messages=messages,
            metadata=row,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> C2CExample:
        return self.rows[index]
