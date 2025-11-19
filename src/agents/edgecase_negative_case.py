from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

from src.core import chat, pick_requirement, parse_json_safely, to_rows_edgecase, write_csv_edgecase, chat_lc
import logging

# Paths (easy-to-change constants for students)
ROOT = Path(__file__).resolve().parents[2]
REQ_DIR = ROOT / "data" / "requirements"  # directory with .txt requirement files
OUT_DIR = ROOT / "outputs" / "edgecase_negative_case_generated"  # where outputs are written
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "test_cases.csv"  # CSV output path
LAST_RAW_JSON = OUT_DIR / "last_raw.json"  # file where raw LLM text is saved

SYSTEM_PROMPT = (ROOT / "src" / "core" / "prompts" / "edgecase_negative_system.txt").read_text(encoding="utf-8")
USER_TEMPLATE = (ROOT / "src" / "core" / "prompts" / "edgecase_negative_user.txt").read_text(encoding="utf-8")

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
    #logger config steps
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

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
    logger.info("Calling chat: provider payload msgs=%d (sys=1,user=1)", len(messages))
    raw = chat_lc(messages)

    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception as e:
        # gentle retry nudge — a pragmatic teaching technique: show how a
        # small reminder can correct common model format mistakes.
        logger.exception(
            "Initial parse_json_safely failed; will nudge and retry. Raw saved at %s",
            LAST_RAW_JSON,
        )
        nudge = (
            raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
        )
        try:
            cases = parse_json_safely(nudge, LAST_RAW_JSON)
        except Exception:
            # Surface a clear runtime error with a pointer to the saved raw
            # output so students can debug model responses during the session.
            logger.error(
                "Could not parse model output after nudge; see %s", LAST_RAW_JSON
            )
            raise RuntimeError(
                f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
            )

    rows = to_rows_edgecase(cases)
    write_csv_edgecase(rows, OUT_CSV)

    logger.info(f"✅ Wrote {len(rows)} test cases to: {OUT_CSV.relative_to(ROOT)}")
    logger.info(f"ℹ️  Raw model output saved at: {LAST_RAW_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

