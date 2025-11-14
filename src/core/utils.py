from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

### Look details of each functions in notion page ###

def pick_requirement(path_arg: str | None, req_dir: Path) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.exists():
            raise FileNotFoundError(f"Requirement file not found: {p}")
        return p
    txts = sorted(Path(req_dir).glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt files found in {req_dir}")
    return txts[0]

### Look details of each functions in notion page ###
def parse_json_safely(text: str, raw_path: Path) -> List[Dict]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding="utf-8")

    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list.")
        return data
    except Exception:
        cleaned = text.strip()
        # If the model wrapped JSON in triple-backtick fences, remove them.
        if cleaned.startswith("```"):
            # Strip surrounding backticks; if a language header is present
            # the first line contains it, so drop that line.
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list after cleanup.")
        return data

### Look details of each functions in notion page ###
def to_rows_edgecase(cases: List[Dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for i, c in enumerate(cases, start=1):
        tid = str(c.get("id") or f"TC-{i:03d}")
        title = str(c.get("title") or "").strip()
        steps_list = c.get("steps") or []
        if not isinstance(steps_list, list):
            steps_list = [str(steps_list)]
        steps = " | ".join(str(s).strip() for s in steps_list if str(s).strip())
        expected = str(c.get("expected") or "").strip()
        priority = str(c.get("priority") or "Medium").strip()
        tags = str(c.get("tags") or "edge").strip()
        likelihood = str(c.get("likelihood") or "Common").strip()
        rows.append([tid, title, steps, expected, priority, tags, likelihood])
    return rows

### Look details of each functions in notion page ###
def write_csv_edgecase(rows: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["TestID", "Title", "Steps", "Expected", "Priority", "Tags", "Likelihood"]
    lines = [",".join(header)]
    for r in rows:
        escaped = [field.replace(",", ";") for field in r]
        lines.append(",".join(escaped))
    path.write_text("\n".join(lines), encoding="utf-8")

### Look details of each functions in notion page ###
def write_json(obj: object, path: Path) -> None:
    """Write an object as pretty JSON to `path`, creating parent dirs.

    Useful shared helper for agents that need to emit JSON outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

### Look details of each functions in notion page ###
def to_rows(cases: List[Dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for i, c in enumerate(cases, start=1):
        tid = str(c.get("id") or f"TC-{i:03d}")
        title = str(c.get("title") or "").strip()
        steps_list = c.get("steps") or []
        if not isinstance(steps_list, list):
            steps_list = [str(steps_list)]
        steps = " | ".join(str(s).strip() for s in steps_list if str(s).strip())
        expected = str(c.get("expected") or "").strip()
        priority = str(c.get("priority") or "Medium").strip()
        rows.append([tid, title, steps, expected, priority])
    return rows

### Look details of each functions in notion page ###
def write_csv(rows: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["TestID", "Title", "Steps", "Expected", "Priority"]
    lines = [",".join(header)]
    for r in rows:
        escaped = [field.replace(",", ";") for field in r]
        lines.append(",".join(escaped))
    path.write_text("\n".join(lines), encoding="utf-8")