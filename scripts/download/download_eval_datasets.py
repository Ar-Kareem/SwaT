#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent.paths import DATASETS_ROOT

DEFAULT_ROOT = DATASETS_ROOT
MMLU_REDUX_REPO = "edinburgh-dawg/mmlu-redux-2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local eval/training dataset snapshots."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root directory. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--mmlu-config",
        action="append",
        default=None,
        help=(
            "MMLU-Redux subject/config to download. Can be passed more than once. "
            "Default: all configs."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["mmlu-redux", "openbookqa", "arc-c", "arc-e", "medqa", "gsm8k", "math-500", "math-algebra",
                 "hellaswag", "winogrande", "mmlu", "medmcqa", "triviaqa", "nq-open"],
        default=None,
        help="Dataset to download. Can be passed more than once. Default: all.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only populate the Hugging Face cache; do not also save load_from_disk snapshots.",
    )
    return parser.parse_args()


def safe_path_part(value: str) -> str:
    return value.replace("/", "__")


def save_snapshot(dataset, snapshot_root: Path, repo_id: str, config: str | None) -> None:
    destination = snapshot_root / safe_path_part(repo_id) / safe_path_part(config or "default")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    hf_home = root / "hf_home"
    hf_datasets_cache = root / "hf_datasets"
    snapshot_root = root / "snapshots"

    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    if not args.cache_only:
        snapshot_root.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install it with: python -m pip install datasets\n"
            "Then rerun this script."
        ) from exc

    print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={hf_datasets_cache}")

    requested = set(args.dataset or ["mmlu-redux", "openbookqa", "arc-c", "medqa", "gsm8k", "math-500", "math-algebra"])

    mmlu_configs = args.mmlu_config
    if "mmlu-redux" in requested and mmlu_configs is None:
        print(f"Fetching configs for {MMLU_REDUX_REPO}")
        mmlu_configs = get_dataset_config_names(
            MMLU_REDUX_REPO,
            cache_dir=str(hf_datasets_cache),
        )

    datasets_to_download = []
    if "mmlu-redux" in requested:
        datasets_to_download.extend((MMLU_REDUX_REPO, config) for config in mmlu_configs)
    if "openbookqa" in requested:
        datasets_to_download.append(("allenai/openbookqa", "main"))
    if "arc-c" in requested:
        datasets_to_download.append(("allenai/ai2_arc", "ARC-Challenge"))
    if "arc-e" in requested:
        datasets_to_download.append(("allenai/ai2_arc", "ARC-Easy"))
    if "medqa" in requested:
        datasets_to_download.append(("GBaker/MedQA-USMLE-4-options", None))
    if "gsm8k" in requested:
        datasets_to_download.append(("openai/gsm8k", "main"))
    if "math-500" in requested:
        datasets_to_download.append(("HuggingFaceH4/MATH-500", None))
    if "math-algebra" in requested:
        datasets_to_download.append(("EleutherAI/hendrycks_math", "algebra"))
    if "hellaswag" in requested:
        datasets_to_download.append(("Rowan/hellaswag", None))
    if "winogrande" in requested:
        datasets_to_download.append(("allenai/winogrande", "winogrande_xl"))
    if "mmlu" in requested:
        datasets_to_download.append(("cais/mmlu", "all"))
    if "medmcqa" in requested:
        datasets_to_download.append(("openlifescienceai/medmcqa", None))
    if "triviaqa" in requested:
        datasets_to_download.append(("mandarjoshi/trivia_qa", "unfiltered.nocontext"))
    if "nq-open" in requested:
        datasets_to_download.append(("google-research-datasets/nq_open", None))

    for repo_id, config in datasets_to_download:
        print(f"Downloading {repo_id} / {config or 'default'}")
        if config is None:
            dataset = load_dataset(repo_id, cache_dir=str(hf_datasets_cache))
        else:
            dataset = load_dataset(repo_id, config, cache_dir=str(hf_datasets_cache))
        split_summary = ", ".join(f"{name}={len(split)}" for name, split in dataset.items())
        print(f"Loaded {repo_id} / {config or 'default'}: {split_summary}")
        if not args.cache_only:
            save_snapshot(dataset, snapshot_root, repo_id, config)

    print("Done.")


if __name__ == "__main__":
    main()
