from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

AVG_SUFFIX = re.compile(r"_avg_\d+$")


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("losses JSON must contain a list of rows")
    if not rows:
        raise ValueError("losses JSON is empty")
    return rows


def is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def numeric_keys(rows: list[dict], x_key: str) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key, value in row.items():
            if key != x_key and is_number(value):
                seen.setdefault(key)
    return list(seen)


def metric_group(key: str) -> str:
    return AVG_SUFFIX.sub("", key.split(":", 1)[0])


def traces(rows: list[dict], x_key: str, keys: list[str]) -> list[dict]:
    out = []
    for key in keys:
        points = [(row[x_key], float(row[key])) for row in rows if is_number(row.get(key)) and is_number(row.get(x_key))]
        if points:
            out.append({
                "x": [x for x, _ in points],
                "y": [y for _, y in points],
                "mode": "lines+markers" if len(points) <= 50 else "lines",
                "name": key,
                "type": "scattergl",
            })
    return out


def stats_table(rows: list[dict], keys: list[str]) -> str:
    parts = ["<table><thead><tr><th>metric</th><th>last</th><th>min</th><th>max</th></tr></thead><tbody>"]
    for key in keys:
        values = [float(row[key]) for row in rows if is_number(row.get(key))]
        parts.append(
            f"<tr><td>{html.escape(key)}</td><td>{values[-1]:.6g}</td><td>{min(values):.6g}</td><td>{max(values):.6g}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def write_html(losses_path: Path, output_path: Path) -> None:
    rows = load_rows(losses_path)
    x_key = "step" if any("step" in row for row in rows) else next(iter(rows[0]))
    keys = numeric_keys(rows, x_key)
    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(metric_group(key), []).append(key)

    sections, scripts = [], []
    for index, (group, members) in enumerate(groups.items()):
        sections.append(f'  <h2>{html.escape(group)}</h2>\n  <div id="plot-{index}" class="plot"></div>')
        scripts.append(
            f'    Plotly.newPlot("plot-{index}", {json.dumps(traces(rows, x_key, members))}, '
            f'{{...layoutBase, yaxis: {{title: {json.dumps(group)}}}}}, {{responsive: true}});'
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(losses_path.name)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111; }}
    h1 {{ font-size: 20px; margin-bottom: 4px; }}
    h2 {{ font-size: 16px; margin-top: 28px; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
    .plot {{ width: 100%; height: 360px; }}
    table {{ border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Training Losses</h1>
  <div class="meta">
    source: {html.escape(str(losses_path))}<br>
    rows: {len(rows)}
  </div>

  <h2>Summary</h2>
  {stats_table(rows, keys)}

{chr(10).join(sections)}

  <script>
    const layoutBase = {{
      margin: {{l: 60, r: 20, t: 20, b: 50}},
      hovermode: "x unified",
      xaxis: {{title: {json.dumps(x_key)}}},
    }};
{chr(10).join(scripts)}
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Plotly HTML dashboard from a training losses.json")
    parser.add_argument("losses_json", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.losses_json.with_suffix(".html")
    write_html(args.losses_json, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
