from typing import List, Dict, TypedDict, Optional

class EdgecaseNegativecaseState(TypedDict, total = False):
    requirement_path: str
    requirements: str
    tests: List[str]
    testrail_case_ids: List[str]

