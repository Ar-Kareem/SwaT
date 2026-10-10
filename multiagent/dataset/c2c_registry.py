from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import ConcatDataset, Dataset, Subset

from multiagent.dataset.c2c_types import C2CExample
from multiagent.dataset.closedbook_qa_dataset import CLOSEDBOOK_TASKS, ClosedBookQADataset
from multiagent.dataset.closedbook_qa_dataset import grade_output as grade_closedbook_output
from multiagent.dataset.hotpotqa_interleaved_dataset import (
    HOTPOTQA_INTERLEAVED_TASK,
    HotpotQAInterleavedDataset,
)
from multiagent.dataset.hotpotqa_interleaved_dataset import grade_output as grade_hotpotqa_interleaved_output
from multiagent.dataset.longbench_qa_dataset import LONG_BENCH_QA_TASK, LongBenchQADataset
from multiagent.dataset.longbench_qa_dataset import grade_output as grade_longbench_qa_output
from multiagent.dataset.math_dataset import DATASET_DIRS as MATH_DATASET_DIRS
from multiagent.dataset.math_dataset import MathDataset, grade_output as grade_math_output
from multiagent.dataset.mcq_dataset import DATASET_DIRS as MCQ_DATASET_DIRS
from multiagent.dataset.mcq_dataset import MCQDataset, grade_output as grade_mcq_output
from multiagent.dataset.mcq_long_dataset import DATASET_DIRS as MCQ_LONG_DATASET_DIRS
from multiagent.dataset.mcq_long_dataset import MCQLongDataset
from multiagent.dataset.phase2_trace import Phase2TraceJsonlDataset
from multiagent.dataset.qa_dataset import QA_TASKS, QADataset, grade_output as grade_qa_output
from multiagent.dataset.qasper_dataset import QASPER_TASKS, QasperDataset, grade_output as grade_qasper_output
from multiagent.dataset.qasper_dataset import QASPER_NO_UNANSWERABLE_OPTION, validate_options as validate_qasper_options
from multiagent.dataset.renderer_continuation import SCHEMA as RENDERER_CONTINUATION_SCHEMA
from multiagent.dataset.renderer_continuation import RendererContinuationJsonlDataset
from multiagent.paths import DATASETS_ROOT


MATH_TASKS = set(MATH_DATASET_DIRS)
MCQ_TASKS = set(MCQ_DATASET_DIRS)
MCQ_LONG_TASKS = set(MCQ_LONG_DATASET_DIRS)
ALL_C2C_TASKS = (
    MATH_TASKS
    | MCQ_TASKS
    | MCQ_LONG_TASKS
    | QA_TASKS
    | CLOSEDBOOK_TASKS
    | QASPER_TASKS
    | {HOTPOTQA_INTERLEAVED_TASK, LONG_BENCH_QA_TASK}
)


@dataclass(frozen=True, slots=True)
class C2CTask:
    spec: str
    dataset: Dataset
    grade: Callable[[str, C2CExample], str | dict[str, Any]]


def parse_c2c_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    if ":" not in spec:
        raise ValueError(f"invalid dataset spec {spec!r}; use task(+key=value)*:split")
    task_expr, split = spec.split(":", 1)
    task, *option_parts = task_expr.split("+")
    options: dict[str, str] = {}
    for part in option_parts:
        key, value = part.split("=", 1) if "=" in part else (part, "")
        if not key:
            raise ValueError(f"invalid dataset option {part!r} in {spec!r}; use +flag or +key=value")
        options[key] = value
    if not task or not split:
        raise ValueError(f"invalid dataset spec {spec!r}; use task(+key=value)*:split")
    if task not in ALL_C2C_TASKS:
        raise KeyError(f"unknown C2C task {task!r}; choose from {sorted(ALL_C2C_TASKS)}")
    return task, split, options


def load_jsonl_dataset(path: Path) -> Dataset:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                first = json.loads(line)
                break
        else:
            raise ValueError(f"empty jsonl dataset {path}")
    if first.get("schema") == RENDERER_CONTINUATION_SCHEMA:
        return RendererContinuationJsonlDataset(path)
    if "trace" in first and "question" in first:
        return Phase2TraceJsonlDataset(path)
    raise ValueError(
        f"unrecognised jsonl row format in {path}: expected schema={RENDERER_CONTINUATION_SCHEMA!r} "
        "(renderer continuation) or question/trace fields (phase-2 trace)"
    )


def limit_dataset(dataset: Dataset, start: int, limit: int | None) -> Dataset:
    if start < 0:
        raise ValueError("start must be non-negative")
    if start >= len(dataset):
        raise ValueError(f"start={start} is outside dataset with {len(dataset)} rows")
    end = len(dataset) if limit is None else min(len(dataset), start + limit)
    return Subset(dataset, range(start, end)) if start or limit is not None else dataset


def load_c2c_dataset_spec(
    spec: str,
    *,
    root: str | Path = DATASETS_ROOT,
    mmlu_subjects: list[str] | None = None,
    include_answer_message: bool = False,
) -> Dataset:
    candidate_path = Path(spec)
    if candidate_path.suffix == ".jsonl" and candidate_path.exists():
        return load_jsonl_dataset(candidate_path)

    task, split, options = parse_c2c_spec(spec)
    if task != HOTPOTQA_INTERLEAVED_TASK and ("limit" in options or "start" in options):
        row_start = int(options.pop("start", "0"))
        row_limit = int(options.pop("limit")) if "limit" in options else None
        inner_spec = task + "".join(f"+{k}={v}" if v else f"+{k}" for k, v in options.items()) + ":" + split
        return limit_dataset(
            load_c2c_dataset_spec(inner_spec, root=root, mmlu_subjects=mmlu_subjects, include_answer_message=include_answer_message),
            row_start,
            row_limit,
        )
    if task == LONG_BENCH_QA_TASK:
        if set(options) != {"manifest"}:
            raise ValueError("use longbench_qa+manifest=PATH:test")
        return LongBenchQADataset(options["manifest"], split=split)
    if task == HOTPOTQA_INTERLEAVED_TASK:
        unknown = set(options) - {"filename", "variant", "start", "limit"}
        if unknown:
            raise ValueError(f"unsupported interleaved HotpotQA options: {sorted(unknown)}")
        variant_value = options.get("variant", "0")
        variant = None if variant_value == "all" else int(variant_value)
        return HotpotQAInterleavedDataset(
            split=split,
            filename=options.get("filename", "hotpotqa_interleaved_28k_validation.jsonl"),
            variant=variant,
            start=int(options.get("start", "0")),
            limit=int(options["limit"]) if "limit" in options else None,
        )
    if task in MATH_TASKS:
        if options:
            raise ValueError(f"unsupported dataset options for {task!r}: {options}")
        return MathDataset(task, split=split, root=root, include_answer_message=include_answer_message)
    if task in MCQ_TASKS:
        if options:
            raise ValueError(f"unsupported dataset options for {task!r}: {options}")
        return MCQDataset(task, split=split, root=root, mmlu_subjects_arg=mmlu_subjects, include_answer_message=include_answer_message)
    if task in MCQ_LONG_TASKS:
        patients_per_context = 5
        ppc_options = [key for key in ("ppc", "patients_per_context") if key in options]
        if task == "longhealth" and len(ppc_options) > 1:
            raise ValueError("use only one of longhealth+ppc=N or +patients_per_context=N")
        if task == "longhealth" and ppc_options:
            ppc_option = ppc_options[0]
            try:
                patients_per_context = int(options[ppc_option])
            except ValueError as error:
                raise ValueError("longhealth ppc must be an integer") from error
            if patients_per_context <= 0:
                raise ValueError("longhealth ppc must be positive")
            options = {key: value for key, value in options.items() if key != ppc_option}
        if options:
            raise ValueError(f"unsupported dataset options for {task!r}: {options}")
        return MCQLongDataset(
            task,
            split=split,
            root=root,
            include_answer_message=include_answer_message,
            patients_per_context=patients_per_context,
        )
    if task in CLOSEDBOOK_TASKS:
        if options:
            raise ValueError(f"unsupported dataset options for {task!r}: {options}")
        return ClosedBookQADataset(task, split=split, root=root, include_answer_message=include_answer_message)
    if task == "qasper":
        validate_qasper_options(options)
        return QasperDataset(
            split=split,
            root=root,
            exclude_unanswerable=QASPER_NO_UNANSWERABLE_OPTION in options,
            length_model=options.get("length_model") or None,
            min_tokens=int(options["min_tokens"]) if "min_tokens" in options else None,
            max_tokens=int(options["max_tokens"]) if "max_tokens" in options else None,
        )
    if task in QA_TASKS:
        unknown = set(options) - {"terse"}
        if unknown:
            raise ValueError(f"unsupported dataset options for {task!r}: {sorted(unknown)}")
        return QADataset(task, split=split, root=root, prompt_style="terse" if "terse" in options else "default")
    raise KeyError(f"unknown C2C dataset spec {spec!r}")


def load_c2c_dataset_specs(
    specs: list[str],
    *,
    root: str | Path = DATASETS_ROOT,
    mmlu_subjects: list[str] | None = None,
    include_answer_message: bool = False,
    start: int = 0,
    limit_per_dataset: int | None = None,
) -> Dataset:
    datasets = [
        limit_dataset(
            load_c2c_dataset_spec(
                spec,
                root=root,
                mmlu_subjects=mmlu_subjects,
                include_answer_message=include_answer_message,
            ),
            start,
            limit_per_dataset,
        )
        for spec in specs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def load_c2c_task(
    spec: str,
    *,
    root: str | Path = DATASETS_ROOT,
    mmlu_subjects: list[str] | None = None,
) -> C2CTask:
    dataset = load_c2c_dataset_spec(
        spec,
        root=root,
        mmlu_subjects=mmlu_subjects,
    )
    task, _, options = parse_c2c_spec(spec)
    if task == HOTPOTQA_INTERLEAVED_TASK:
        grade_fn = grade_hotpotqa_interleaved_output
    elif task == LONG_BENCH_QA_TASK:
        grade_fn = grade_longbench_qa_output
    elif task in MATH_TASKS:
        grade_fn = grade_math_output
    elif task in MCQ_TASKS or task in MCQ_LONG_TASKS:
        grade_fn = grade_mcq_output
    elif task in CLOSEDBOOK_TASKS:
        grade_fn = grade_closedbook_output
    elif task in QA_TASKS:
        grade_fn = grade_qa_output
    elif task in QASPER_TASKS:
        grade_fn = grade_qasper_output
    else:
        raise KeyError(f"no grader for C2C task {spec!r}")

    def grade(output: str, example: C2CExample) -> str | dict[str, Any]:
        return grade_fn(output, example.metadata)

    return C2CTask(spec=spec, dataset=dataset, grade=grade)
