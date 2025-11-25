import logging
from pathlib import Path
from typing import List

from .state import EdgecaseNegativecaseState
from src.core import pick_requirement, chat_lc, parse_json_safely, to_rows_edgecase, write_csv_edgecase
from src.integrations.testrail import map_case_to_testrail_payload, create_case, add_result

logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")
logger = logging.getLogger(__name__)

"""
1. Requirement path 
2. Prompt path
3. output path
"""

# ---------------- Path Setup ----------------
# nodes.py is at: <project-root>/src/graph/test_case_generator/nodes.py
# Step up 3 levels → <project-root>
ROOT = Path(__file__).resolve().parents[3]

# Data folders (relative to project root)
REQ_DIR = ROOT / "data" / "requirements"
OUT_DIR = ROOT / "outputs" / "testcase_generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUT_CSV = OUT_DIR / "test_cases.csv"
LAST_RAW_JSON = OUT_DIR / "last_raw.json"

# Prompt files (inside src/core/prompts)
PROMPTS_DIR = ROOT / "src" / "core" / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "testcase_system.txt").read_text(encoding="utf-8")
USER_PROMPT_TEMPLATE = (PROMPTS_DIR / "testcase_user.txt").read_text(encoding="utf-8")


def read_requirements(state: EdgecaseNegativecaseState) -> EdgecaseNegativecaseState:
    req_path = pick_requirement(None, REQ_DIR)
    logger.info(f"📄 Reading requirements from {req_path.name}")
    state["requirements"] = req_path.read_text(encoding="utf-8").strip()
    return state

def generate_edgecase_negative_cases(state: EdgecaseNegativecaseState) -> EdgecaseNegativecaseState:
    logger.info("🤖 Generating test cases with LLM...")
    user_prompt = USER_PROMPT_TEMPLATE.format(requirement_text = state.get("requirements", ""))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    raw = chat_lc(messages)
    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception:
        logger.warning("⚠️ Could not parse JSON from LLM, writing raw output")
        cases = []
    rows = to_rows_edgecase(cases)
    write_csv_edgecase(rows, OUT_CSV)
    logger.info(f"✅ Wrote {len(rows)} test cases to {OUT_CSV}")

    state["tests"] = [c.get("title", "Untitled test") for c in cases]
    return state

def push_to_testrail(state: EdgecaseNegativecaseState) -> EdgecaseNegativecaseState:
    logger.info("📤 Pushing test cases to TestRail...")
    tests = state.get("tests", [])
    if not tests:
        logger.warning("⚠️ No test cases to push to TestRail.")
        return state
    
    created_ids: List[int] = []
    for title in tests:
        payload = map_case_to_testrail_payload({"title":title})
        try:
            res = create_case(payload)
            cid = res.get("id")
            if cid:
                created_ids.append(cid)
                add_result(cid, status_id=3, comment="Seeded by LangGraph pipeline")   
        except Exception as e:
            logger.error(f"❌ Failed to create TestRail case '{title}': {e}")

    state["testrail_case_ids"] = [str(cid) for cid in created_ids]
    logger.info(f"✅ Created {len(created_ids)} TestRail cases: {created_ids}")


