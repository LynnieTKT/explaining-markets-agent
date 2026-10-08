"""What information explains the abnormal return beyond the earnings surprise?

Runs entirely on the archived data (no LLM calls). Each section prints one
table. Percentile targets are computed within quarter, like the official scorer.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from examples.scoring import add_percentiles, outcomes_frame, score_submission  # noqa: E402

ARCHIVE = Path(__file__).resolve().parent.parent / "examples" / "data" / "archive"
QUARTERS = ["2025Q4", "2026Q1", "2026Q2", "2026Q3"]
BASELINES = {
    "deepseek/ea-explain-contemp-summary": "deepseek",
    "gemini/ea-explain-contemp-summary": "gemini",
    "moonshotai/ea-explain-contemp-summary": "kimi",
    "openai/ea-explain-contemp-summary": "gpt5nano",
    "z-ai/ea-explain-contemp-summary": "glm",
}
BCOLS = list(BASELINES.values())


def load(q: str) -> list[dict]:
    out = []
    with gzip.open(ARCHIVE / f"EARNINGS_RELEASE_{q}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "scored" and r.get("disclosure"):
                out.append(r)
    return out


def items(r: dict) -> dict:
    return {it["id"]: it.get("content") for it in r["disclosure"].get("items", [])}


def build(q: str) -> pd.DataFrame:
    recs = load(q)
    frame = outcomes_frame(recs)
    by_id = {r["event_id"]: r for r in recs}
    frame = frame[frame["surprise"].notna()].reset_index(drop=True)
    rows = []
    for eid, tk in zip(frame["event_id"], frame["identifier_value"], strict=True):
        r = by_id[eid]
        it = items(r)
        bp = r.get("baseline_predictions") or {}
        row = {k: (bp.get(key) or {}).get(tk) for key, k in BASELINES.items()}
        row["hour_utc"] = int(r["event_datetime"][11:13])
        facts = it.get("earnings-call-facts") or []
        row["facts_text"] = " ".join(facts) if isinstance(facts, list) else ""
        row["n_facts"] = len(facts) if isinstance(facts, list) else 0
        row["has_preview"] = int(bool(it.get("earnings-preview")))
        st = it.get("option-implied-stats") or {}

        def val(k: str) -> float | None:
            b = st.get(k) if isinstance(st, dict) else None
            return b.get("value") if isinstance(b, dict) and b.get("status") == "ok" else None

        row["impl_vol"] = val("implied_earnings_volatility")
        row["impl_move"] = val("implied_absolute_earnings_move")
        row["skew"] = val("skew_25_delta")
        rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(rows)], axis=1)
    frame["quarter"] = q
    frame = add_percentiles(frame)
    frame["avg5"] = frame[BCOLS].mean(axis=1)
    return frame


def resid_after_surprise(f: pd.DataFrame) -> np.ndarray:
    x = np.c_[np.ones(len(f)), f["surprise_pct"].to_numpy()]
    beta, *_ = np.linalg.lstsq(x, f["y"].to_numpy(), rcond=None)
    return f["y"].to_numpy() - x @ beta


def d_r2(f: pd.DataFrame, col: str) -> float:
    s = score_submission(f, col)
    return s["delta_r_squared_imputed"] if s["delta_r_squared_imputed"] is not None else float("nan")


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


KEYWORDS = {
    "guid_raise": r"guidance.{0,40}(raised|increased|lifted|boosted)|(raised|raises|raising|lifted|boosted).{0,40}guidance|raised.{0,30}(outlook|forecast)",
    "guid_cut": r"guidance.{0,40}(lowered|cut|reduced|trimmed|withdrawn|withdrew)|(lowered|cut|reduced|trimmed|withdrew).{0,40}(guidance|outlook|forecast)",
    "guid_reaffirm": r"(reaffirmed|reiterated|maintained|unchanged).{0,40}(guidance|outlook|forecast)",
    "beat": r"\b(beat|beating|exceeded|exceeding|above consensus|ahead of (?:consensus|estimates|expectations)|surpass)",
    "miss": r"\b(missed|missing|below consensus|below (?:estimates|expectations)|short of|fell short)",
    "record": r"\brecord\b",
    "decline": r"\b(declined|decrease|decreased|fell|down \d|lower than)",
    "growth_strong": r"(grew|increased|up|growth of|rose)\s(?:by\s)?\d{2,}%",
    "margin_up": r"margin.{0,40}(expanded|improved|increased|up \d)",
    "margin_down": r"margin.{0,40}(contracted|declined|decreased|compressed|down \d|pressure)",
    "buyback": r"(buyback|repurchase)",
    "weak_demand": r"(weak|soft|softness|headwind|challenging|uncertain)",
    "tariff": r"tariff",
    "layoff": r"(layoff|restructuring|workforce reduction|headcount reduction)",
    "acquisition": r"(acquisition|acquire|merger)",
}


def keyword_features(text: pd.Series) -> pd.DataFrame:
    low = text.str.lower()
    return pd.DataFrame({k: low.str.contains(p, regex=True).astype(int) for k, p in KEYWORDS.items()})


def main() -> None:
    frames = {q: build(q) for q in QUARTERS}
    allf = pd.concat(frames.values(), ignore_index=True)

    section("1. How much does the earnings surprise alone explain, per quarter")
    rows = []
    for q, f in frames.items():
        s = score_submission(f, "avg5")
        rows.append(
            {
                "quarter": q,
                "n": len(f),
                "R2_surprise": s["r_squared_surprise_imputed"],
                "corr(surprise_pct,y)": f["surprise_pct"].corr(f["y"]),
                **{f"dR2_{c}": d_r2(f, c) for c in ["glm", "gemini", "gpt5nano", "avg5"]},
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    section("2. Is the surprise relation linear? mean y by surprise decile (all quarters)")
    allf["surp_dec"] = pd.qcut(allf["surprise_pct"], 10, labels=False, duplicates="drop")
    print(
        allf.groupby("surp_dec")
        .agg(n=("y", "size"), mean_y=("y", "mean"), mean_avg5=("avg5", "mean"))
        .to_string(float_format=lambda v: f"{v:.3f}")
    )

    section("3. Where do the LLM baselines add value? corr(avg5, residual) by surprise tercile")
    rows = []
    for q, f in frames.items():
        e = resid_after_surprise(f)
        f = f.assign(resid=e, terc=pd.qcut(f["surprise_pct"], 3, labels=["low", "mid", "high"]))
        for t, g in f.groupby("terc", observed=True):
            rows.append({"quarter": q, "surprise": t, "n": len(g), "corr(avg5,resid)": g["avg5"].corr(g["resid"]), "corr(glm,resid)": g["glm"].corr(g["resid"])})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    section("4. Baseline calibration: mean realized y by avg5 bucket (all quarters)")
    allf["avg5_bin"] = pd.cut(allf["avg5"], [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    print(
        allf.groupby("avg5_bin", observed=True)
        .agg(n=("y", "size"), mean_y=("y", "mean"), sd_y=("y", "std"))
        .to_string(float_format=lambda v: f"{v:.3f}")
    )
    print("\navg5 distribution:", allf["avg5"].describe().round(3).to_dict())

    section("5. Timing: pre-market vs after-close events (all quarters)")
    allf["session"] = np.where(allf["hour_utc"] < 15, "pre-market", "after-close")
    rows = []
    for (q, s), g in allf.groupby(["quarter", "session"]):
        rows.append({"quarter": q, "session": s, "n": len(g), "sd_car1": g["car1"].std(), "dR2_avg5": d_r2(g.reset_index(drop=True).pipe(lambda d: add_percentiles(d.drop(columns=["y", "surprise_pct"]))), "avg5")})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    section("6. Option-implied stats (2026Q3 only): do they explain magnitude or direction?")
    f3 = frames["2026Q3"].copy()
    f3["resid"] = resid_after_surprise(f3)
    have = f3[f3["impl_move"].notna()].copy()
    print(f"events with option stats: {len(have)} of {len(f3)}")
    print("corr(impl_move, |car1|)        =", round(have["impl_move"].corr(have["car1"].abs()), 3))
    print("corr(impl_move, |resid|)       =", round(have["impl_move"].corr(have["resid"].abs()), 3))
    print("corr(impl_move, resid)         =", round(have["impl_move"].corr(have["resid"]), 3))
    sk = have[have["skew"].notna()]
    print(f"events with skew: {len(sk)};  corr(skew, resid) =", round(sk["skew"].corr(sk["resid"]), 3), " corr(skew, y) =", round(sk["skew"].corr(sk["y"]), 3))
    # Experiment: scale the LLM view by the implied move (bigger expected move -> more extreme prediction)
    for c in ["avg5", "glm"]:
        dev = f3[c] - f3[c].mean()
        mv = f3["impl_move"].fillna(f3["impl_move"].median())
        f3[f"{c}_x_move"] = 0.5 + dev * (mv / mv.median())
        print(f"dR2 {c}: {d_r2(f3, c):.4f}  ->  scaled by implied move: {d_r2(f3, f'{c}_x_move'):.4f}")
    # Does having a preview / option stats coincide with better LLM accuracy?
    for flag in ["has_preview"]:
        for v, g in f3.groupby(flag):
            print(f"{flag}={v}: n={len(g)}, corr(avg5,resid)={g['avg5'].corr(g['resid']):.3f}")

    section("7. Which facts keywords carry information beyond the surprise? (train Q4'25+Q1+Q2, test Q3)")
    kw = keyword_features(allf["facts_text"])
    allf = pd.concat([allf, kw], axis=1)
    allf["resid"] = np.concatenate([resid_after_surprise(frames[q]) for q in QUARTERS])
    train = allf[allf["quarter"] != "2026Q3"]
    test = allf[allf["quarter"] == "2026Q3"]
    rows = []
    for k in KEYWORDS:
        g = train.groupby(k)["resid"].agg(["size", "mean"])
        if 1 not in g.index:
            continue
        rows.append({"keyword": k, "freq": g.loc[1, "size"] / len(train), "resid_if_present": g.loc[1, "mean"], "resid_if_absent": g.loc[0, "mean"], "corr_train": train[k].corr(train["resid"]), "corr_test": test[k].corr(test["resid"])})
    print(pd.DataFrame(rows).sort_values("corr_train", key=abs, ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Linear model on keywords, out of sample
    X = np.c_[np.ones(len(train)), train[list(KEYWORDS)].to_numpy()]
    beta, *_ = np.linalg.lstsq(X, train["resid"].to_numpy(), rcond=None)
    Xt = np.c_[np.ones(len(test)), test[list(KEYWORDS)].to_numpy()]
    f3 = frames["2026Q3"].copy()
    f3["kw_model"] = Xt @ beta
    f3["glm_plus_kw"] = f3["glm"].fillna(f3["glm"].mean()) + 2 * f3["kw_model"]
    print(f"\nkeyword-only model dR2 on Q3: {d_r2(f3, 'kw_model'):.4f}")
    print(f"glm dR2: {d_r2(f3, 'glm'):.4f}   glm + keyword model: {d_r2(f3, 'glm_plus_kw'):.4f}")

    section("8. Baseline disagreement: is the average worse when the five models disagree? (Q3)")
    f3 = frames["2026Q3"].copy()
    f3["resid"] = resid_after_surprise(f3)
    f3["spread"] = f3[BCOLS].std(axis=1)
    f3["spread_bin"] = pd.qcut(f3["spread"], 3, labels=["agree", "mid", "disagree"])
    print(f3.groupby("spread_bin", observed=True).apply(lambda g: pd.Series({"n": len(g), "corr(avg5,resid)": g["avg5"].corr(g["resid"]), "sd_avg5": g["avg5"].std()})).to_string(float_format=lambda v: f"{v:.3f}"))

    section("9. Out-of-sample recalibration of avg5: decile->mean_y mapping fit on earlier quarters, applied to Q3")
    train = allf[allf["quarter"] != "2026Q3"]
    edges = np.quantile(train["avg5"].dropna(), np.linspace(0, 1, 11))
    edges[0], edges[-1] = -1, 2
    train_bin = pd.cut(train["avg5"], edges, labels=False, include_lowest=True)
    mapping = train.groupby(train_bin)["y"].mean()
    f3["avg5_recal"] = pd.cut(f3["avg5"], edges, labels=False, include_lowest=True).map(mapping)
    print(f"avg5 dR2 on Q3: {d_r2(f3, 'avg5'):.4f}   recalibrated: {d_r2(f3, 'avg5_recal'):.4f}")


if __name__ == "__main__":
    main()
