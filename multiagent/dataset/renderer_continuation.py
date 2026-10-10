from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from multiagent.dataset.c2c_types import C2CExample


SCHEMA = "multikv.renderer_continuation.v1"


class RendererContinuationJsonlDataset(Dataset):

    def __init__(self, path: str | Path):
        self.path = Path(path)
        rows: list[C2CExample] = []
        with self.path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != SCHEMA:
                    raise ValueError(
                        f"unsupported renderer-continuation schema in {self.path}:{line_number}: "
                        f"{row.get('schema')!r}"
                    )
                prefix = str(row["prefix_text"])
                continuation = str(row["continuation_text"])
                raw_prefix = [{"role": "raw", "content": prefix}]
                rows.append(
                    C2CExample(
                        id=str(row["id"]),
                        dataset=str(row["dataset"]),
                        split=str(row.get("split", "train")),
                        phase1_messages=raw_prefix,
                        sender_messages=raw_prefix,
                        receiver_messages=[],
                        target_text=continuation,
                        prompt_mode="raw_continuation",
                        messages=None,
                        metadata=row,
                    )
                )
        if not rows:
            raise ValueError(f"renderer continuation dataset is empty: {self.path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> C2CExample:
        return self.rows[index]
