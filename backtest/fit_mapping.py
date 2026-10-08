"""Fit the structured-view -> percentile map on early-quarter events, test on the contest window.

Usage: uv run python fit_mapping.py --label structured-nano-q3 [--split 2026-08-10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as rb
from examples.scoring import add_percentiles, outcomes_frame, score_submission
from predict import MAPPING_WEIGHTS, SurpriseView, map_to_percentile, view_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="structured-nano-q3")
    ap.add_argument("--split", default="2026-08-10")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    recs = rb.load_records("2026Q3", "2026-07-01", args.end)
    cache: dict[str, dict] = {}
    for line in (Path("cache") / f"{args.label}.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            cache[d["event_id"]] = d
    recs = [r for r in recs if r["event_id"] in cache and cache[r["event_id"]].get("view")]
    print(f"events with structured view: {len(recs)}")

    frame = outcomes_frame(recs)
    frame = frame[frame["surprise"].notna()].reset_index(drop=True)
    by_id = {r["event_id"]: r for r in recs}
    views = {eid: SurpriseView(**cache[eid]["view"]) for eid in frame["event_id"]}
    feats = pd.DataFrame([view_features(views[eid]) for eid in frame["event_id"]])
    frame = pd.concat([frame, feats], axis=1)
    frame["date"] = [by_id[e]["event_datetime"][:10] for e in frame["event_id"]]
    for key, col in rb.BASELINE_KEYS.items():
        frame[col] = [
            (by_id[e].get("baseline_predictions") or {}).get(key, {}).get(t)
            for e, t in zip(frame["event_id"], frame["identifier_value"], strict=True)
        ]
    frame["hand"] = [map_to_percentile(views[e]) for e in frame["event_id"]]
    frame["expected_only"] = [views[e].expected_reaction for e in frame["event_id"]]

    train = frame[frame["date"] < args.split].copy()
    test = frame[frame["date"] >= args.split].copy()
    train = add_percentiles(train)
    test = add_percentiles(test)
    print(f"train: {len(train)} events (< {args.split}), test: {len(test)} events (contest window)")

    names = list(MAPPING_WEIGHTS)
    # Target: the residual of y after the surprise is what the contest rewards, but
    # at prediction time we do not observe the surprise. Fit on y directly and on
    # the residual; report both.
    x = np.c_[np.ones(len(train)), train[names].to_numpy()]
    xs = np.c_[np.ones(len(train)), train["surprise_pct"].to_numpy()]
    bs, *_ = np.linalg.lstsq(xs, train["y"].to_numpy(), rcond=None)
    resid = train["y"].to_numpy() - xs @ bs
    fits = {}
    for target_name, target in [("y", train["y"].to_numpy()), ("resid", resid)]:
        lam = args.ridge * np.eye(x.shape[1])
        lam[0, 0] = 0.0
        beta = np.linalg.solve(x.T @ x + lam, x.T @ target)
        fits[target_name] = dict(zip(["_intercept", *names], beta, strict=True))

    def apply(f: pd.DataFrame, w: dict[str, float]) -> pd.Series:
        return 0.5 + sum(w[k] * f[k] for k in names)

    for name, w in fits.items():
        train[f"fit_{name}"] = apply(train, w)
        test[f"fit_{name}"] = apply(test, w)

    # Feature-by-feature contribution (test window): drop one at a time from the y-fit.
    rows = []
    for col in ["hand", "expected_only", "fit_y", "fit_resid", *rb.BASELINE_KEYS.values()]:
        rows.append(
            {
                "submission": col,
                "dR2_train": score_submission(train, col)["delta_r_squared_imputed"],
                "dR2_test": score_submission(test, col)["delta_r_squared_imputed"],
            }
        )
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + pd.DataFrame(rows).sort_values("dR2_test", ascending=False).to_string(index=False))

    print("\nfitted weights (target=y):")
    for k, v in fits["y"].items():
        print(f"  {k:16s} {v:+.4f}")

    print("\nablation on test (y-fit refit without each feature):")
    base = score_submission(test, "fit_y")["delta_r_squared_imputed"]
    for drop in names:
        keep = [n for n in names if n != drop]
        xk = np.c_[np.ones(len(train)), train[keep].to_numpy()]
        lam = args.ridge * np.eye(xk.shape[1])
        lam[0, 0] = 0.0
        b = np.linalg.solve(xk.T @ xk + lam, xk.T @ train["y"].to_numpy())
        w = dict(zip(keep, b[1:], strict=True))
        test["_abl"] = 0.5 + sum(w[k] * test[k] for k in keep)
        d = score_submission(test, "_abl")["delta_r_squared_imputed"]
        print(f"  without {drop:16s} {d:.4f}  ({d - base:+.4f})")

    # Blends with the baselines, for reference only (we cannot call those models live).
    test["fit_y+glm"] = (test["fit_y"] + test["baseline-glm"].fillna(test["baseline-glm"].mean())) / 2
    print(f"\nreference blend fit_y + glm on test: {score_submission(test, 'fit_y+glm')['delta_r_squared_imputed']:.4f}")

    out = Path(f"weights_{args.label}.json")
    out.write_text(json.dumps({k: round(v, 4) for k, v in fits["y"].items() if k != "_intercept"}, indent=2))
    print(f"\nweights written to {out}")


if __name__ == "__main__":
    sys.exit(main())
