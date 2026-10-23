"""Replays all six replayable Token Factory runs and verifies gateway header reproduction (T2).

Implements Section 10 (T2) of `docs/specs/TERMINAL_APP_SPEC.md`:
- Across the six replayable runs (B, C, E, F, G v2, G v2.1), verifies that today's condenser
  reproduces 3,185 of 3,239 recorded requests (`per_turn_tool_chars_before/after`) exactly.
- Confirms that all 54 differences contain a `[TIMEOUT]` log in runs recorded before `78df5c0`:
  B: 8 of 600, E: 19 of 600, F: 27 of 564, and 0 in C (486), G v2 (459), G v2.1 (530).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tui.data import DEFAULT_RESULTS_DIR, verify_all_replay_headers

EXPECTED_PER_RUN = {
    "B": (592, 600, 8),
    "C": (486, 486, 0),
    "E": (581, 600, 19),
    "F": (537, 564, 27),
    "G_v2": (459, 459, 0),
    "G_v2_1": (530, 530, 0),
}


def main() -> None:
    report = verify_all_replay_headers(results_dir=DEFAULT_RESULTS_DIR)
    print(json.dumps(report, indent=2))

    if (
        report["total_requests"] != 3239
        or report["matched_requests"] != 3185
        or report["diff_requests"] != 54
        or report["timeout_diff_requests"] != 54
        or report["unexpected_diffs"]
    ):
        sys.exit(1)

    for run_key, (exp_match, exp_total, exp_diff) in EXPECTED_PER_RUN.items():
        r = report["per_run"].get(run_key, {})
        if (
            r.get("matched_requests") != exp_match
            or r.get("total_requests") != exp_total
            or r.get("diff_requests") != exp_diff
            or r.get("timeout_diff_requests") != exp_diff
        ):
            sys.exit(1)


if __name__ == "__main__":
    main()
