from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

from src.core import chat, pick_requirement, parse_json_safely, to_rows_edgecase, write_csv_edgecase

# Paths (easy-to-change constants for students)
ROOT = Path(__file__).resolve().parents[2]
REQ_DIR = ROOT / "data" / "requirements"  # directory with .txt requirement files
OUT_DIR = ROOT / "outputs" / "edgecase_negative_case_generated"  # where outputs are written
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "test_cases.csv"  # CSV output path
LAST_RAW_JSON = OUT_DIR / "last_raw.json"  # file where raw LLM text is saved

SYSTEM_PROMPT = """You are a senior QA assistant.
From the user template/requirement content prepare concise edge cases and negative cases.
Think step-by-step about the requirement and produce ONLY a JSON array of
test cases using this schema:

[
  {
    "id": "TC-001",
    "title": "Short test title",
    "steps": ["step 1", "step 2"],
    "expected": "Expected result",
    "priority": "High|Medium|Low",
    "tags" : ["edge" | "negative" | "happy"],
    "likelihood": "Rare|Uncommon|Common|Frequent"
  }
]

Rules:
- Return JSON ONLY (no prose, no fences).
- Provide at least 6 and at most 12 cases. At least 30% must be tagged as edge or negative.
- Steps should be short, imperative, and precise.
"""

USER_TEMPLATE = 'Requirement:\n"""{requirement_text}"""'  # user content injected
# `USER_TEMPLATE` wraps the requirement text so the model sees a clear input
# block; we keep it simple for students to inspect and modify.

Message = Dict[str, str]

def main() -> None:
    """Run the testcase agent end-to-end.

    Flow:
    1. Pick a requirement file (CLI arg or first file in `data/requirements`).
    2. Read the requirement text.
    3. Build a system + user message pair and call `chat(messages)`.
    4. Save raw model output to `outputs/last_raw.json` and parse it as JSON.
    5. Convert parsed cases to CSV rows and write `outputs/test_cases.csv`.

    Error handling and teaching hooks:
    - We save the raw model text to `LAST_RAW_JSON` so students can inspect
      model failures or formatting issues.
    - If parsing fails, we perform a single "nudge" retry that reminds the
      model to return pure JSON. If it still fails we raise a helpful error
      pointing to the raw file.
    - The agent is deliberately thin: it focuses on orchestration. Students
      can extend it later to add retries, rate-limiting, human-in-the-loop
      review pages, or direct integrations with Jira/TestRail.
    """

    req_path = pick_requirement(sys.argv[1] if len(sys.argv) > 1 else None, REQ_DIR)
    requirement_text = req_path.read_text(encoding="utf-8").strip()

    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(requirement_text=requirement_text),
        },
    ]

    # Call the LLM via the provider-agnostic `chat` function. The returned
    # `raw` is the assistant's text; for Day-1 we expect the model to return
    # a pure JSON array (see SYSTEM_PROMPT) so downstream parsing is simple.
    raw = chat(messages)

    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception as e:
        # gentle retry nudge — a pragmatic teaching technique: show how a
        # small reminder can correct common model format mistakes.
        nudge = (
            raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
        )
        try:
            cases = parse_json_safely(nudge, LAST_RAW_JSON)
        except Exception:
            # Surface a clear runtime error with a pointer to the saved raw
            # output so students can debug model responses during the session.
            raise RuntimeError(
                f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
            )

    rows = to_rows_edgecase(cases)
    write_csv_edgecase(rows, OUT_CSV)

    print(f"✅ Wrote {len(rows)} test cases to: {OUT_CSV.relative_to(ROOT)}")
    print(f"ℹ️  Raw model output saved at: {LAST_RAW_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

