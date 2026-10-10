#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.paths import DATASETS_ROOT

DEFAULT_ROOT = DATASETS_ROOT


DATASETS = {
    "2wikimqa": {
        "repo_id": "voidful/2WikiMultihopQA",
        "files": {
            "train": "train.json",
            "validation": "dev.json",
            "test": "test.json",
        },
        "config": "default",
    },
    "hotpotqa": {
        "repo_id": "hotpotqa/hotpot_qa",
        "config": "distractor",
    },
    "musique": {
        "repo_id": "voidful/MuSiQue",
        "files": {
            "train": "musique_ans_v1.0_train.jsonl",
            "validation": "musique_ans_v1.0_dev.jsonl",
            "test": "musique_ans_v1.0_test.jsonl",
        },
        "config": "default",
    },
    "qasper": {
        "kind": "qasper",
        "repo_id": "allenai/qasper",
        "config": None,
    },
    "quality": {
        "kind": "quality_github",
        "repo_url": "https://github.com/nyu-mll/quality-public",
        "name": "quality",
    },
    "longhealth": {
        "kind": "longhealth_github",
        "repo_url": "https://github.com/kbressem/LongHealth",
        "name": "longhealth",
    },
    "race": {
        "repo_id": "ehovy/race",
        "config": "all",
    },
    "squad": {
        "repo_id": "rajpurkar/squad",
        "config": "plain_text",
    },
    "narrativeqa": {
        "repo_id": "deepmind/narrativeqa",
        "config": "default",
    },
    "boolq": {
        "repo_id": "google/boolq",
        "config": "default",
    },
    "cmmlu": {"kind": "hf_snapshot", "repo_id": "haonan-li/cmmlu", "extract_zips": True},
    "ceval": {"kind": "hf_snapshot", "repo_id": "ceval/ceval-exam"},
    "mmmlu": {"kind": "hf_snapshot", "repo_id": "openai/MMMLU"},
}


def main() -> None:
    parser = base_parser("Download supported Hugging Face QA datasets.")
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    parser.add_argument("--name", help="Snapshot directory name. Defaults to the selected dataset name.")
    args = parser.parse_args()

    if args.dataset == "all" and args.name:
        parser.error("--name cannot be used with dataset=all")

    keys = DATASETS if args.dataset == "all" else [args.dataset]
    failures = []
    for key in keys:
        spec = DATASETS[key]
        print(f"\n== {key} ==")
        try:
            if spec.get("kind") == "hf_snapshot":
                download_snapshot(spec["repo_id"], args.name, args.root, args.cache_only, spec.get("extract_zips", False))
            elif spec.get("kind") == "quality_github":
                download_quality(spec["repo_url"], args.name or spec["name"], args.root, args.cache_only)
            elif spec.get("kind") == "longhealth_github":
                download_longhealth(spec["repo_url"], args.name or spec["name"], args.root, args.cache_only)
            elif spec.get("kind") == "qasper":
                download_qasper(
                    spec["repo_id"],
                    spec["config"],
                    args.name,
                    args.root,
                    args.cache_only,
                )
            elif "files" in spec:
                download_json_files(
                    spec["repo_id"],
                    spec["files"],
                    spec["config"],
                    args.name,
                    args.root,
                    args.cache_only,
                )
            else:
                download_dataset(
                    spec["repo_id"],
                    spec["config"],
                    args.name,
                    args.root,
                    args.cache_only,
                )
        except Exception as exc:
            if args.dataset != "all":
                raise
            failures.append((key, exc))
            print(f"FAILED {key}: {type(exc).__name__}: {exc}")

    if failures:
        failed = ", ".join(key for key, _ in failures)
        raise SystemExit(f"failed datasets: {failed}")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root. Default: $MULTIKV_DATASETS_ROOT.",
    )
    parser.add_argument("--cache-only", action="store_true")
    return parser


def configure_cache(root: Path) -> tuple[Path, Path, Path]:
    hf_home = root / "hf_home"
    hf_datasets = root / "hf_datasets"
    snapshots = root / "snapshots"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)
    return hf_home, hf_datasets, snapshots


def safe_path_part(value: str) -> str:
    return value.replace("/", "__")


def snapshot_path(snapshots: Path, repo_id: str, config: str | None, name: str | None = None) -> Path:
    if name:
        return snapshots / name
    parts = [safe_path_part(repo_id)]
    if config:
        parts.append(safe_path_part(config))
    return snapshots.joinpath(*parts)


def download_snapshot(repo_id: str, name: str | None, root: Path, cache_only: bool, extract_zips: bool) -> None:
    import zipfile

    from huggingface_hub import snapshot_download

    hf_home, _, snapshots = configure_cache(root)
    target = snapshot_path(snapshots, repo_id, None, name)
    if cache_only:
        print(f"Cache only: {repo_id} -> {target}")
        return
    print(f"Downloading snapshot {repo_id} -> {target}")
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(target), cache_dir=str(hf_home / "hub"))
    if extract_zips:
        for archive in sorted(target.glob("*.zip")):
            print(f"Extracting {archive.name}")
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
    print(f"Saved {target}")


def download_dataset(repo_id: str, config: str | None, name: str | None, root: Path, cache_only: bool) -> None:
    root = root.resolve()
    hf_home, hf_datasets, snapshots = configure_cache(root)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        raise SystemExit("Missing dependency: datasets. Install it with: python -m pip install datasets") from exc

    print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={hf_datasets}")
    print(f"Downloading {repo_id}" + (f" / {config}" if config else ""))

    dataset = load_dataset(repo_id, config, cache_dir=str(hf_datasets)) if config else load_dataset(repo_id, cache_dir=str(hf_datasets))
    print("Loaded splits: " + ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()))

    if cache_only:
        return

    destination = snapshot_path(snapshots, repo_id, config, name)
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


def _arrow_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _load_json_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def download_json_files(
    repo_id: str,
    files: dict[str, str],
    config: str | None,
    name: str | None,
    root: Path,
    cache_only: bool,
) -> None:
    root = root.resolve()
    hf_home, hf_datasets, snapshots = configure_cache(root)

    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={hf_datasets}")
    print(f"Downloading raw JSON files from {repo_id}")

    paths = {
        split: Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
        for split, filename in files.items()
    }
    if cache_only:
        return

    dataset = DatasetDict()
    for split, path in paths.items():
        rows = [{key: _arrow_safe(value) for key, value in row.items()} for row in _load_json_records(path)]
        dataset[split] = Dataset.from_list(rows)
    print("Loaded splits: " + ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()))

    destination = snapshot_path(snapshots, repo_id, config, name)
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using cached file: {destination}")
        return destination
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        with open(destination, "wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
    return destination


def _qasper_rows_from_archive(archive_path: Path, filename: str) -> list[dict]:
    with tarfile.open(archive_path, "r:gz") as archive:
        member = next((item for item in archive.getmembers() if Path(item.name).name == filename), None)
        if member is None:
            raise FileNotFoundError(f"{filename} not found in {archive_path}")
        f = archive.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"{filename} could not be extracted from {archive_path}")
        data = json.load(f)
    return [dict(row, id=id_) for id_, row in data.items()]


def download_qasper(repo_id: str, config: str | None, name: str | None, root: Path, cache_only: bool) -> None:
    root = root.resolve()
    hf_home, hf_datasets, snapshots = configure_cache(root)

    try:
        from datasets import Dataset, DatasetDict
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    cache_dir = hf_datasets / "qasper"
    train_dev = _download_file(
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        cache_dir / "qasper-train-dev-v0.3.tgz",
    )
    test = _download_file(
        "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
        cache_dir / "qasper-test-and-evaluator-v0.3.tgz",
    )
    print(f"HF_HOME={hf_home}")
    print(f"HF_DATASETS_CACHE={hf_datasets}")

    if cache_only:
        return

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(_qasper_rows_from_archive(train_dev, "qasper-train-v0.3.json")),
            "validation": Dataset.from_list(_qasper_rows_from_archive(train_dev, "qasper-dev-v0.3.json")),
            "test": Dataset.from_list(_qasper_rows_from_archive(test, "qasper-test-v0.3.json")),
        }
    )
    print("Loaded splits: " + ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()))
    destination = snapshot_path(snapshots, repo_id, config, name)
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


def clone_repo(repo_url: str, workdir: Path) -> Path:
    destination = workdir / Path(repo_url).stem
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destination)], check=True)
    return destination


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def quality_rows(path: Path) -> list[dict]:
    rows = []
    for row in load_jsonl(path):
        questions = []
        for question in row.get("questions", []):
            item = {
                "question": question.get("question", ""),
                "question_unique_id": question.get("question_unique_id"),
                "options": question.get("options", []),
                "difficult": question.get("difficult"),
            }
            if "gold_label" in question:
                item["gold_label"] = question["gold_label"]
            questions.append(item)
        rows.append(
            {
                "id": row.get("article_id", row.get("set_unique_id")),
                "article_id": row.get("article_id"),
                "set_unique_id": row.get("set_unique_id"),
                "title": row.get("title"),
                "source": row.get("source"),
                "article": row.get("article", ""),
                "questions": questions,
            }
        )
    return rows


def download_quality(repo_url: str, name: str, root: Path, cache_only: bool) -> None:
    root = root.resolve()
    _, _, snapshots = configure_cache(root)

    try:
        from datasets import Dataset, DatasetDict
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        repo = clone_repo(repo_url, Path(tmp))
        if cache_only:
            return
        data_dir = repo / "data" / "v1.0.1"
        files = {
            "train": data_dir / "QuALITY.v1.0.1.htmlstripped.train",
            "dev": data_dir / "QuALITY.v1.0.1.htmlstripped.dev",
            "test": data_dir / "QuALITY.v1.0.1.htmlstripped.test",
        }
        dataset = DatasetDict({split: Dataset.from_list(quality_rows(path)) for split, path in files.items()})

    print("Loaded splits: " + ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()))
    destination = snapshots / name
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


def longhealth_context(texts: dict) -> str:
    return "\n\n".join(f"{key}\n{text}" for key, text in sorted(texts.items()))


def longhealth_rows(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    rows = []
    for patient_id, patient in data.items():
        questions = []
        for question in patient.get("questions", []):
            questions.append(
                {
                    "question": question.get("question", ""),
                    "answer_a": question.get("answer_a"),
                    "answer_b": question.get("answer_b"),
                    "answer_c": question.get("answer_c"),
                    "answer_d": question.get("answer_d"),
                    "answer_e": question.get("answer_e"),
                    "correct": question.get("correct"),
                    "No": question.get("No"),
                }
            )
        rows.append(
            {
                "id": patient_id,
                "name": patient.get("name"),
                "birthday": patient.get("birthday"),
                "diagnosis": patient.get("diagnosis"),
                "context": longhealth_context(patient.get("texts", {})),
                "questions": questions,
            }
        )
    return rows


def download_longhealth(repo_url: str, name: str, root: Path, cache_only: bool) -> None:
    root = root.resolve()
    _, _, snapshots = configure_cache(root)

    try:
        from datasets import Dataset, DatasetDict
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    with tempfile.TemporaryDirectory() as tmp:
        repo = clone_repo(repo_url, Path(tmp))
        if cache_only:
            return
        rows = longhealth_rows(repo / "data" / "benchmark_v5.json")
        dataset = DatasetDict({"test": Dataset.from_list(rows)})

    print("Loaded splits: " + ", ".join(f"{split}={len(rows)}" for split, rows in dataset.items()))
    destination = snapshots / name
    dataset.save_to_disk(str(destination))
    print(f"Saved snapshot: {destination}")


if __name__ == "__main__":
    main()
