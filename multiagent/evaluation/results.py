from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from multiagent.dataset.c2c_types import C2CExample, c2c_example_to_dict


EVAL_JSON_TEXT_LIMIT = 1000


EVAL_JSON_TEXT_EDGE = 500


EVAL_JSON_OMITTED_ROW_FIELDS = (
    "construction",
    "context",
    "distractor_source_ids",
    "hop1",
    "hop2",
    "messages",
    "phase1_messages",
    "receiver_messages",
    "sender_messages",
    "supporting_facts",
)


class EvalJsonCompactor:

    def __init__(self, tokenizer, limit: int = EVAL_JSON_TEXT_LIMIT, edge: int = EVAL_JSON_TEXT_EDGE):
        if edge * 2 > limit:
            raise ValueError("JSON compaction edge length cannot exceed half the token limit")
        self.tokenizer = tokenizer
        self.limit = limit
        self.edge = edge
        self._text_cache: dict[str, str] = {}

    def compact_text(self, text: str) -> str:
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= self.limit:
            compacted = text
        else:
            omitted = len(token_ids) - 2 * self.edge
            prefix = self.tokenizer.decode(token_ids[: self.edge], skip_special_tokens=False)
            suffix = self.tokenizer.decode(token_ids[-self.edge :], skip_special_tokens=False)
            compacted = (
                f"{prefix}\n\n[... {omitted} tokens omitted from saved evaluation JSON; "
                "evaluation used the complete text ...]\n\n"
                f"{suffix}"
            )
        self._text_cache[text] = compacted
        return compacted

    def compact(self, value):
        if isinstance(value, str):
            return self.compact_text(value)
        if isinstance(value, dict):
            return {key: self.compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.compact(item) for item in value]
        if isinstance(value, tuple):
            return [self.compact(item) for item in value]
        return value


def average_metrics(metric_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        name: row["sum"] / row["count"]
        for name, row in metric_stats.items()
        if row["count"]
    }


def _format_metric_name(name: str) -> str:
    return name.replace("_", "")


def score_postfix(stats: dict[str, dict[str, int]], metric_stats: dict[str, dict[str, dict[str, float]]]) -> dict[str, str]:
    postfix = {}
    total_correct = 0
    total_seen = 0
    total_metric_stats: dict[str, dict[str, float]] = {}
    for task, row in stats.items():
        seen = row["seen"]
        correct = row["correct"]
        total_correct += correct
        total_seen += seen
        task_metric_text = ",".join(
            f"{_format_metric_name(name)}={value:.3f}"
            for name, value in average_metrics(metric_stats.get(task, {})).items()
        )
        postfix[task] = f"{correct}/{seen}" if seen else "0/0"
        if task_metric_text:
            postfix[task] += f",{task_metric_text}"
        for metric_name, metric_row in metric_stats.get(task, {}).items():
            total_row = total_metric_stats.setdefault(metric_name, {"sum": 0.0, "count": 0.0})
            total_row["sum"] += metric_row["sum"]
            total_row["count"] += metric_row["count"]
    all_metric_text = ",".join(
        f"{_format_metric_name(name)}={value:.3f}"
        for name, value in average_metrics(total_metric_stats).items()
    )
    postfix["all"] = f"{total_correct}/{total_seen}" if total_seen else "0/0"
    if all_metric_text:
        postfix["all"] += f",{all_metric_text}"
    return postfix


def sample_input_text(sample: C2CExample) -> str:
    for message in sample.messages or sample.phase1_messages:
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def normalize_grade_result(result: str | dict[str, Any]) -> tuple[str, dict[str, float], dict[str, Any]]:
    if isinstance(result, dict):
        label = str(result.get("grade", "fail"))
        metrics = {
            key: float(value)
            for key, value in result.items()
            if key != "grade" and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return label, metrics, result
    return result, {}, {"grade": result}


def update_metric_stats(metric_stats: dict[str, dict[str, dict[str, float]]], task: str, metrics: dict[str, float]) -> None:
    task_metrics = metric_stats.setdefault(task, {})
    for name, value in metrics.items():
        row = task_metrics.setdefault(name, {"sum": 0.0, "count": 0.0})
        row["sum"] += float(value)
        row["count"] += 1.0


def print_eval_sample(position: int, total: int, task: str, sample: C2CExample, output: str, result: str, metrics: dict[str, float]):
    print("\n" + "=" * 80)
    print(f"sample {position}/{total} | task={task} | id={sample.id}")
    print("\ninput:")
    print(sample_input_text(sample))
    print("\noutput:")
    print(output)
    print(f"\ngold: {sample.metadata.get('answer')}")
    metric_text = ", ".join(f"{name}={value:.3f}" for name, value in metrics.items())
    print(f"grade: {result}" + (f" ({metric_text})" if metric_text else ""))
    print("=" * 80 + "\n", flush=True)


def compact_eval_sample(sample: C2CExample, json_compactor: EvalJsonCompactor) -> dict[str, Any]:
    payload = c2c_example_to_dict(sample)
    for field in EVAL_JSON_OMITTED_ROW_FIELDS:
        payload.pop(field, None)
    return json_compactor.compact(payload)


def eval_result_overview(data: dict[str, Any]) -> dict[str, Any] | None:
    summary = data.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("all"), dict):
        return None
    overall = summary["all"]
    accuracy = overall.get("accuracy")
    metrics = overall.get("metrics") or {}
    failed = sum(
        int(task_summary.get("fail", 0) or 0)
        for task, task_summary in summary.items()
        if task != "all" and isinstance(task_summary, dict)
    )
    return {
        "mode": data.get("mode"),
        "tasks": data.get("tasks", []),
        "samples_evaluated": int(overall.get("seen", 0) or 0),
        "correct": int(overall.get("correct", 0) or 0),
        "accuracy_percent": round(100.0 * float(accuracy), 6) if accuracy is not None else None,
        "metrics_percent": {
            name: round(100.0 * float(value), 6)
            for name, value in metrics.items()
        },
        "failed_grades": failed,
        "stop_reasons": overall.get("stop_reasons", {}),
    }


def eval_json_format_metadata() -> dict[str, Any]:
    return {
        "layout": "human-readable metadata followed by one compact JSON object per result row",
        "omitted_row_fields": list(EVAL_JSON_OMITTED_ROW_FIELDS),
        "omitted_content_remains_in_source_dataset": True,
        "text_compaction": {
            "threshold_tokens": EVAL_JSON_TEXT_LIMIT,
            "prefix_tokens": EVAL_JSON_TEXT_EDGE,
            "suffix_tokens": EVAL_JSON_TEXT_EDGE,
            "evaluation_uses_full_text": True,
        },
    }


def prepare_eval_json(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    payload.pop("result_overview", None)
    if isinstance(payload.get("rows"), list):
        payload["rows"] = [
            {key: value for key, value in row.items() if key not in EVAL_JSON_OMITTED_ROW_FIELDS}
            for row in payload["rows"]
        ]
    overview = eval_result_overview(payload)
    return {"result_overview": overview, **payload} if overview is not None else payload


def save_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = prepare_eval_json(data)
    rows = payload.pop("rows", None)
    with open(tmp_path, "w", encoding="utf-8") as f:
        if rows is None:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        else:
            metadata = json.dumps(payload, indent=2, ensure_ascii=False)
            f.write(metadata[:-1].rstrip())
            f.write(',\n  "rows": [\n')
            for position, row in enumerate(rows):
                f.write("    ")
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write(",\n" if position + 1 < len(rows) else "\n")
            f.write("  ]\n}\n")
    tmp_path.replace(path)


def human_elapsed(seconds: float) -> str:
    for unit, divisor in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if seconds >= divisor:
            return f"{seconds / divisor:.1f}{unit}"
    return f"{seconds:.1f}s"


def elapsed_metadata(started_at: float) -> dict[str, Any]:
    elapsed = time.time() - started_at
    return {"elapsed_seconds": round(elapsed, 3), "elapsed_human": human_elapsed(elapsed)}


def _json_safe_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_config_value(item) for item in value]
    return str(value)


def evaluation_config_metadata(args, *, dtype: torch.dtype, script, command_line=None) -> dict[str, Any]:
    return {
        "script": str(Path(script).resolve()),
        "command_line": [str(value) for value in (sys.argv if command_line is None else command_line)],
        "arguments": {
            key: _json_safe_config_value(value)
            for key, value in sorted(vars(args).items())
        },
        "resolved_dtype": str(dtype).removeprefix("torch."),
        "input_truncation": {"max_length": args.max_length, "side": "context_prefix"},
    }


def summarize(
    stats: dict[str, dict[str, int]],
    stop_stats: dict[str, dict[str, int]],
    metric_stats: dict[str, dict[str, dict[str, float]]],
) -> dict:
    summary = {}
    total_seen = 0
    total_correct = 0
    total_stop_reasons: dict[str, int] = {}
    total_metric_stats: dict[str, dict[str, float]] = {}
    for task, row in stats.items():
        seen = row["seen"]
        correct = row["correct"]
        total_seen += seen
        total_correct += correct
        for reason, count in stop_stats.get(task, {}).items():
            total_stop_reasons[reason] = total_stop_reasons.get(reason, 0) + count
        for metric_name, metric_row in metric_stats.get(task, {}).items():
            total_row = total_metric_stats.setdefault(metric_name, {"sum": 0.0, "count": 0.0})
            total_row["sum"] += metric_row["sum"]
            total_row["count"] += metric_row["count"]
        summary[task] = {
            **row,
            "accuracy": correct / seen if seen else None,
            "stop_reasons": stop_stats.get(task, {}),
            "metrics": average_metrics(metric_stats.get(task, {})),
        }
    summary["all"] = {
        "seen": total_seen,
        "correct": total_correct,
        "accuracy": total_correct / total_seen if total_seen else None,
        "stop_reasons": total_stop_reasons,
        "metrics": average_metrics(total_metric_stats),
    }
    return summary
