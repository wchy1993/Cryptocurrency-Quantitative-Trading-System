from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mtf_htf_diagnostics import diagnose
from scripts.mtf_htf_summary import summarize


def summarize_stage(payload: dict[str, Any]) -> dict[str, Any]:
    experiments = []
    for item in payload.get("experiments", []):
        report = item.get("report", {})
        experiments.append(
            {
                "name": item.get("name"),
                "config": item.get("config"),
                "summary": item.get("summary") or summarize(report),
                "diagnostics": item.get("diagnostics") or diagnose(report),
            }
        )
    output = {
        key: payload.get(key)
        for key in ("trade_start", "trade_end", "cost_experiment", "backtest_mode", "symbols")
    }
    output["experiments"] = experiments
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact staged MTF report")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    compact = summarize_stage(payload)
    Path(args.output).write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
