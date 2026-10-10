#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent.paths import MODELS_ROOT


def args():
    p = argparse.ArgumentParser()
    p.add_argument("repo_id", nargs="?")
    p.add_argument("--root", type=Path, default=MODELS_ROOT)
    p.add_argument("--ignore", action="append", default=[], help="Extra glob to skip (e.g. consolidated.safetensors for Mistral repos that ship both formats)")
    return p.parse_args()


a = args()
org, name = a.repo_id.split("/", 1)
target = a.root / org.lower() / name
target.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=a.repo_id,
    local_dir=str(target),
    ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "tf_model.*", "flax_model.*", *a.ignore],
)

print(target)
