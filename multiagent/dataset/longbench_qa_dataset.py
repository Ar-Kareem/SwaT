from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample
from multiagent.dataset.hotpotqa_interleaved_dataset import (
    context_messages, full_messages, question_messages,
)
from multiagent.dataset.qa_dataset import squad_exact_match, squad_f1


LONG_BENCH_QA_TASK = "longbench_qa"


def grade_output(output: str, sample: dict) -> dict:
    answers = sample["answers"]
    em = squad_exact_match(output, answers)
    return {"grade": "correct" if em else "incorrect", "em": em,
            "f1": squad_f1(output, answers)}


class LongBenchQADataset(Dataset):
    preserve_order = True

    def __init__(self, manifest: str, split: str = "test"):
        if split != "test":
            raise ValueError("LongBench QA manifests use the released test split")
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        ids = [row["id"] for row in self.rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("manifest must contain nonempty, unique sample IDs")
        for row in self.rows:
            if row["source_dataset"] != "musique" or not row["answers"]:
                raise ValueError("expected labeled MuSiQue rows")
            if hashlib.sha256(row["context"].encode()).hexdigest() != row["context_sha256"]:
                raise ValueError(f"context checksum mismatch: {row['id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        context, question = row["context"], row["question"]
        messages = full_messages(context, question)
        return C2CExample(
            id=row["id"], dataset=LONG_BENCH_QA_TASK, split="test",
            phase1_messages=messages, sender_messages=context_messages(context),
            receiver_messages=question_messages(question), target_text=row["answers"][0],
            prompt_mode="split_prompt", messages=messages,
            metadata={**row, "answer": row["answers"][0], "context_id": row["id"]},
        )
