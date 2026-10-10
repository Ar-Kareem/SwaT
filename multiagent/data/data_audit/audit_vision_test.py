import json
from pathlib import Path
import re


def short_id(value):
    value = str(value)
    match = re.fullmatch(r"(?:test|train|validation)-([0-9]+)-of-[0-9]+_([0-9]+)", value)
    if match:
        return f"{int(match[1])}:{int(match[2])}"
    match = re.fullmatch(r"(human|augmented)_([0-9]+)", value)
    if match:
        return f"{match[1][0]}{int(match[2])}"
    match = re.fullmatch(r"VizWiz_(?:test|train|val)_([0-9]+)\.jpg", value)
    if match:
        return str(int(match[1]))
    match = re.fullmatch(r"(?:test|train|dev)_([0-9]+)", value)
    if match:
        return str(int(match[1]))
    match = re.fullmatch(r"(?:test|validation|dev)_([A-Za-z_]+)_([0-9]+)", value)
    if match:
        return f"{match[1]}:{match[2]}"
    match = re.fullmatch(r"([0-9]+)\.png-([0-9]+)", value)
    if match:
        return f"{match[1]}:{match[2]}"
    return value


def write_audit(split, output):
    source = Path(__file__).resolve().parent.parent / split
    inputs = sorted(source.glob("*/questions.json"))
    if not inputs:
        raise FileNotFoundError(f"No question files found in {source}")
    lines = []
    for path in inputs:
        seen = set()
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            sample_id = short_id(row["question_id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate compact ID in {path.parent.name}: {sample_id}")
            seen.add(sample_id)
            lines.append(json.dumps([path.parent.name, sample_id], separators=(",", ":")))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{output.name}: {len(lines):,} samples, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    write_audit("vision_test", Path(__file__).resolve().with_suffix(".jsonl"))
