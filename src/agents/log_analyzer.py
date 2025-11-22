from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple
import re
import json
import argparse

from src.core import chat,chat_lc
from src.core.utils import write_json
from langchain_core.prompts import PromptTemplate
import logging

#part-1 Parsing & Grouping

def load_logs(paths: Iterable[Path]) -> Iterable[str]:
    """Yield lines from given log file paths."""
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line.rstrip("\n")


def parse_log_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Parse 'YYYY-MM-DD HH:MM:SS [LEVEL] message' into (ts, level, msg)."""
    pattern = (
        r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s+\[(?P<level>INFO|WARN|ERROR)\]\s+"
        r"(?P<msg>.*)"
    )
    m = re.match(pattern, line)
    if not m:
        return None
    return m.group("ts"), m.group("level"), m.group("msg").strip()


def compute_signature(msg: str, max_tokens: int = 4) -> str:
    """Very small normalization to form a short signature."""
    s = msg.lower()
    s = re.sub(r"/[-\w./]+", " ", s)    # strip paths
    s = re.sub(r"\d+", " ", s)          # strip numbers
    s = re.sub(r"[^a-z\s]", " ", s)     # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    return " ".join(tokens[:max_tokens]) if tokens else msg[:32]


def group_events(lines: Iterable[str]) -> list[dict]:

    """Aggregate parsed log lines into groups keyed by signature."""
    groups: dict[str, dict] = {}
    for line in lines:
        parsed = parse_log_line(line)
        if not parsed:
            continue
        _, level, msg = parsed
        sig = compute_signature(msg)
        g = groups.get(sig)
        if not g:
            g = {
                "signature": sig,
                "count": 0,
                "levels": {"INFO": 0, "WARN": 0, "ERROR": 0},
                "examples": [],
            }
            groups[sig] = g
        g["count"] += 1
        g["levels"][level] = g["levels"].get(level, 0) + 1
        if len(g["examples"]) < 3:
            g["examples"].append(line)

    return sorted(groups.values(), key=lambda x: x["count"], reverse=True)

#part-2 

def build_llm_messages(groups: list, total_events: int, top_n: int = 3) -> list:
    """Construct system+user messages for the LLM."""
    payload = groups[:top_n]  # we’ll pass top_n computed in main (can be 'all')

    # small hinting: surface exception-like tokens from examples
    for g in payload:
        exs = []
        for ex_line in g.get("examples", []):
            for f in re.findall(r"([A-Za-z_]+(?:Error|Exception))", ex_line):
                if f not in exs:
                    exs.append(f)
        if exs:
            g["exceptions"] = exs

    #read prompts 
    ROOT = Path(__file__).resolve().parents[2]
    PROMPTS_DIR = ROOT / "src" / "core" / "prompts"

    system = (PROMPTS_DIR / "log_system.txt").read_text(encoding="utf-8")
    user_template_str = (PROMPTS_DIR / "log_user.txt").read_text(encoding="utf-8")
    user_template = PromptTemplate.from_template(user_template_str)

    user_payload = json.dumps({"groups": payload, "total_events": total_events}, indent=2)

    user =  user_template.format(payload_json=user_payload)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_llm(messages: list, timeout: int) -> str:
    """Thin wrapper over core chat client (returns raw string)."""
    return chat_lc(messages, timeout=timeout)


# def parse_llm_output(raw: str) -> dict:
#     """Parse LLM JSON or save raw to outputs/log_analyzer/last_raw.json and raise."""
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         out_dir = Path("outputs") / "log_analyzer"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         (out_dir / "last_raw.json").write_text(raw, encoding="utf-8")
#         raise RuntimeError(
#             f"LLM output was not valid JSON. Raw saved to {out_dir / 'last_raw.json'}"
#         )
def parse_llm_output(raw: str) -> dict:
    """Parse JSON output from LLM, stripping code fences and ignoring leading/trailing text.

    If parsing fails, save the raw output to outputs/log_analyzer/last_raw.json and raise.
    """
    if not raw or not raw.strip():
        raise RuntimeError("LLM returned empty response")

    # 1️⃣ Strip Markdown code fences (```json ... ```)
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.I)
    cleaned = cleaned.strip()

    # 2️⃣ Try parsing the cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 3️⃣ Try to extract the first valid JSON object/array from inside text
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 4️⃣ Still invalid → save raw for debugging
        out_dir = Path("outputs") / "log_analyzer"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "last_raw.json").write_text(raw, encoding="utf-8")
        raise RuntimeError(
            f"LLM output was not valid JSON. Raw saved to {out_dir / 'last_raw.json'}"
        )
    
#part-3 CLI Entry

def main(argv: Optional[list] = None) -> None:
    #logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=600, help="LLM timeout seconds")
    parser.add_argument(
        "--llm-top",
        type=int,
        default=3,
        help="How many top groups to send to LLM (use -1 for ALL)",
    )
    args = parser.parse_args(argv)

    # parse & group
    paths = [Path(p) for p in args.inputs]
    lines = list(load_logs(paths))
    logger.info("Read %d lines from inputs=%s", len(lines), paths)
    logger.debug("First 3 lines: %s", lines[:3])
    groups = group_events(lines)
    logger.info("Grouped into %d signatures", len(groups))
    logger.debug("Top signatures: %s", [g["signature"] for g in groups[:5]])
    total = sum(g["count"] for g in groups)

    # Option A (1): send ALL groups when --llm-top = -1
    llm_top = len(groups) if args.llm_top < 0 else args.llm_top
    messages = build_llm_messages(groups, total, top_n=llm_top)

    logger.info("Calling LLM with top_n=%d (total_events=%d)", args.llm_top, total)
    logger.debug(
        "LLM payload size=%d chars",
        len(json.dumps({"groups": groups[: args.llm_top], "total_events": total})),
    )

    # call LLM and parse
    raw = call_llm(messages, timeout=args.timeout)
    findings = parse_llm_output(raw)

    # Ensure summary dict exists
    if "summary" not in findings or not isinstance(findings.get("summary"), dict):
        findings["summary"] = {}

    # Option A (2): HARD-MERGE LLM groups back onto ALL parsed groups
    llm_by_sig = {
        g.get("signature"): g
        for g in findings.get("groups", [])
        if isinstance(g, dict) and "signature" in g
    }
    merged_groups = []
    for g in groups:
        sig = g["signature"]
        base = {
            "signature": sig,
            "count": g.get("count", 0),
            "levels": g.get("levels", {}),
            "examples": g.get("examples", []),
        }
        enrich = llm_by_sig.get(sig, {})
        base["probable_root_cause"] = enrich.get("probable_root_cause", "")
        base["recommendation"] = enrich.get("recommendation", "")
        merged_groups.append(base)
    findings["groups"] = merged_groups

    # Option A (3): ALWAYS compute error_rate locally from parsed groups
    local_errors = sum(g.get("levels", {}).get("ERROR", 0) for g in groups)
    computed_error_rate = round(local_errors / max(1, total), 3)
    findings["summary"]["total_events"] = total
    findings["summary"]["error_rate"] = computed_error_rate

        # --- Override top_signatures and short_summary deterministically ---
    total_info = sum(g.get("levels", {}).get("INFO", 0) for g in groups)
    total_warn = sum(g.get("levels", {}).get("WARN", 0) for g in groups)
    total_err  = sum(g.get("levels", {}).get("ERROR", 0) for g in groups)

    # Choose top 3 signatures prioritizing ERRORs, then WARNs, then INFO
    def sigs_by(level):
        return sorted(
            [(g["signature"], g["levels"].get(level, 0)) for g in groups if g["levels"].get(level, 0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )

    top = [s for s, _ in sigs_by("ERROR")]
    if len(top) < 3:
        top += [s for s, _ in sigs_by("WARN") if s not in top]
    if len(top) < 3:
        top += [s for s, _ in sigs_by("INFO") if s not in top]
    top = top[:3]

    findings["summary"]["top_signatures"] = top
    findings["summary"]["short_summary"] = (
        f"{total} events → {total_err} errors, {total_warn} warnings, {total_info} info "
        f"(error rate {computed_error_rate:.0%})."
        + (f" Top errors: {', '.join(top)}." if total_err else "")
    )

    # Small helper: if a group lacks root cause, try exception tokens from its examples
    for g in findings["groups"]:
        pr = g.get("probable_root_cause", "")
        rec = g.get("recommendation", "")
        if not pr:
            exs = []
            for ex_line in g.get("examples", []):
                for f in re.findall(r"([A-Za-z_]+(?:Error|Exception))", ex_line):
                    if f not in exs:
                        exs.append(f)
            if exs:
                g["probable_root_cause"] = ", ".join(exs)
                if not rec:
                    g["recommendation"] = f"Investigate {exs[0]} and related services"

    # write outputs
    out_dir = Path("outputs") / "log_analyzer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "log_findings.json"
    out_md = out_dir / "log_summary.md"
    write_json(findings, out_json)
    out_md.write_text(
        "# Summary\n" + json.dumps(findings.get("summary", {}), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
