from __future__ import annotations

import re
from typing import Any

import torch

from multiagent.paths import REPO_ROOT
from pathlib import Path


TRANSFER_FILES = ("phase2_transfer.pt", "phase1_transfer.pt")
CHECKPOINT_RE = re.compile(r"^(checkpoint|exit-checkpoint)-(\d+)$")


def checkpoint_step(path: str | Path) -> int | None:
    match = CHECKPOINT_RE.match(Path(path).name)
    return int(match.group(2)) if match else None


def largest_checkpoint_dir(path: str | Path) -> Path:
    path = Path(path)
    candidates = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is None or not any((child / name).exists() for name in TRANSFER_FILES):
            continue
        priority = 1 if child.name.startswith("exit-checkpoint-") else 0
        candidates.append((step, priority, child))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint-* or exit-checkpoint-* directories with transfer weights found in {path}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def resolve_checkpoint_dir(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {path}")
    if any((path / name).exists() for name in TRANSFER_FILES):
        return path
    return largest_checkpoint_dir(path)


def checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        checkpoint_dir = resolve_checkpoint_dir(path)
        for name in TRANSFER_FILES:
            candidate = checkpoint_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no phase2_transfer.pt or phase1_transfer.pt found in {checkpoint_dir}")
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    return path


def load_checkpoint_state(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = checkpoint_file(path)
    state = torch.load(path, map_location="cpu", mmap=True)
    return path, state


def resolve_referenced_checkpoint(path: str | Path, referring_checkpoint: Path) -> Path:
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            REPO_ROOT / path,
            REPO_ROOT / "multiagent" / path,
            referring_checkpoint.parent / path,
        ])
    for candidate in candidates:
        if candidate.exists():
            return checkpoint_file(candidate)
    raise FileNotFoundError(
        f"checkpoint {referring_checkpoint} references missing phase1 checkpoint {path!s}"
    )


def c2c_model_names(state: dict[str, Any], state_path: Path) -> tuple[str, str]:
    saved_args = state.get("args", {})
    sender_name = saved_args.get("sender_model")
    receiver_name = saved_args.get("receiver_model")
    if sender_name and receiver_name:
        return str(sender_name), str(receiver_name)

    phase1_reference = state.get("phase1_checkpoint") or saved_args.get("phase1_checkpoint")
    if phase1_reference:
        phase1_path = resolve_referenced_checkpoint(phase1_reference, state_path)
        _, phase1_state = load_checkpoint_state(phase1_path)
        phase1_args = phase1_state.get("args", {})
        sender_name = phase1_args.get("sender_model")
        receiver_name = phase1_args.get("receiver_model")
        if sender_name and receiver_name:
            return str(sender_name), str(receiver_name)

    raise ValueError(
        f"C2C checkpoint {state_path} does not identify sender_model and receiver_model"
    )
