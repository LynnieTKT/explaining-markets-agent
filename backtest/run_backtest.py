"""Backtest predict.py against the historical archive with the official scorer.

Usage:
    uv run python run_backtest.py [--quarter 2026Q3] [--start 2026-08-10] [--end 2026-09-30]
                                  [--workers 8] [--limit N] [--label starter-nano]

Predictions are cached in cache/<label>.jsonl keyed by event_id, so re-runs
only call the LLM for events not yet predicted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter-modal"
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "src"))
sys.path.insert(0, str(EXAMPLES / "src"))

load_dotenv(STARTER / ".env")

from examples.scoring import add_percentiles, outcomes_frame, score_submission  # noqa: E402
from predict import _ask_llm, predict_structured  # noqa: E402

BASELINE_KEYS = {
    "deepseek/ea-explain-contemp-summary": "baseline-deepseek",
    "gemini/ea-explain-contemp-summary": "baseline-gemini",
    "moonshotai/ea-explain-contemp-summary": "baseline-kimi",
    "openai/ea-explain-contemp-summary": "baseline-gpt5nano",
    "z-ai/ea-explain-contemp-summary": "baseline-glm",
}


def load_records(quarter: str, start: str, end: str) -> list[dict]:
    path = EXAMPLES / "data" / "archive" / f"EARNINGS_RELEASE_{quarter}.jsonl.gz"
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "scored" or not r.get("disclosure"):
                continue
            if start <= r["event_datetime"][:10] <= end:
                out.append(r)
    return out


def load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            cache[d["event_id"]] = d["pred"]
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", default="2026Q3")
    ap.add_argument("--start", default="2026-08-10")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="starter-nano")
    ap.add_argument("--strategy", choices=["baseline", "structured"], default="baseline")
    args = ap.parse_args()

    records = load_records(args.quarter, args.start, args.end)
    if args.limit:
        records = records[: args.limit]
    print(f"events in window: {len(records)}")

    cache_path = Path(__file__).parent / "cache" / f"{args.label}.jsonl"
    cache_path.parent.mkdir(exist_ok=True)
    cache = load_cache(cache_path)
    todo = [r for r in records if r["event_id"] not in cache]
    print(f"cached: {len(records) - len(todo)}, to predict: {len(todo)}")

    def work(r: dict) -> tuple[str, float, dict | None]:
        ticker = r["focal_assets"][0]["identifier_value"]
        if args.strategy == "structured":
            p, view = predict_structured(summary=r["disclosure"], ticker=ticker, event_type=r["event_type"])
            return r["event_id"], float(p), (view.model_dump() if view is not None else None)
        p = _ask_llm(summary=r["disclosure"], ticker=ticker, event_type=r["event_type"])
        return r["event_id"], float(p), None

    t0 = time.time()
    with cache_path.open("a") as out, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                eid, p, view = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {exc}")
                continue
            cache[eid] = p
            out.write(json.dumps({"event_id": eid, "pred": p, "view": view}) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f}s")

    frame = outcomes_frame(records)
    for key, col in BASELINE_KEYS.items():
        frame[col] = [
            (r.get("baseline_predictions") or {}).get(key, {}).get(r["focal_assets"][0]["identifier_value"])
            for r in records
        ]
    frame[args.label] = [cache.get(r["event_id"]) for r in records]
    frame = add_percentiles(frame)

    rows = []
    for col in [args.label, *BASELINE_KEYS.values()]:
        s = score_submission(frame, col)
        rows.append(
            {
                "submission": col,
                "n_obs": s["n_obs"],
                "R2_surprise": s["r_squared_surprise_imputed"],
                "R2_full": s["r_squared_imputed"],
                "dR2_common": s["delta_r_squared_imputed"],
                "dR2_own": s["delta_r_squared"],
            }
        )
    table = pd.DataFrame(rows).sort_values("dR2_common", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print()
    print(table.to_string(index=False))
    print(f"\nour preds: mean={frame[args.label].mean():.3f} sd={frame[args.label].std():.3f}")
    table.to_csv(Path(__file__).parent / f"results_{args.label}.csv", index=False)


if __name__ == "__main__":
    main()
