"""Frozen historical correction of the two-model forecast.

The release artifact carries both training-data and selection-data availability
times. An event cannot use the correction until its own knowledge cutoff covers
both. Current-event realized returns and true earnings surprises are never inputs.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
import gzip
import hashlib
import json
import os
import re

import httpx
import numpy as np

from explaining_markets.market_features import chart_frame, features_from_frames

ARTIFACT_DIR = Path(__file__).with_name("stat_model")
PRICE_TIMEOUT_SECONDS = 6.0
SUBMISSION_RESERVE_SECONDS = 15.0
MODELS = ("gpt-5-mini", "gpt-5.4-nano")
TOPICS = {
    "bank": r"net interest|deposits|loan portfolio|credit losses",
    "clinical": r"clinical|\bfda\b|patients|phase [123i]",
    "software": r"subscription|annual recurring|\barr\b|software|cloud",
    "hardware": r"semiconductor|wafer|chip|data center|networking",
    "property": r"\baffo\b|\bffo\b|occupancy|rental|leased",
    "energy": r"barrels|natural gas|crude|drilling",
    "retail": r"same-store|comparable sales|restaurant|retail stores",
    "industrial": r"manufacturing|industrial|aerospace|defense",
    "utility": r"rate base|megawatt|gigawatt|electricity",
}


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("A timezone is required")
    return parsed


def available_for_event(metadata: dict, event: dict) -> bool:
    cutoff = timestamp(event["knowledge_cutoff"])
    required = max(timestamp(metadata[k]) for k in
                   ("training_data_available_at", "selection_data_available_at"))
    return cutoff >= required


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads((ARTIFACT_DIR / "metadata.json").read_text())


@lru_cache(maxsize=1)
def load_artifacts():
    import joblib
    metadata = load_metadata()
    path = ARTIFACT_DIR / "correction.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Correction artifact does not match its metadata")
    history_path = ARTIFACT_DIR / "history.json.gz"
    if hashlib.sha256(history_path.read_bytes()).hexdigest() != metadata["history_sha256"]:
        raise ValueError("History artifact does not match its metadata")
    with gzip.open(history_path, "rt") as source:
        history = json.load(source)
    return joblib.load(path), history


@lru_cache(maxsize=64)
def fetch_prices(symbol: str, cutoff: str):
    end = timestamp(cutoff)
    response = httpx.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe=""),
        params={"period1": int((end-timedelta(days=540)).timestamp()),
                "period2": int(end.timestamp()), "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=PRICE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json().get("chart", {}).get("result") or []
    if not rows:
        raise ValueError("Daily prices unavailable")
    frame = chart_frame(rows[0], symbol)
    if frame is None:
        raise ValueError("Daily prices have an unsupported symbol or timezone")
    return frame


def build_features(event: dict, summary: dict, ticker: str, views: dict,
                   history: dict, stock, market):
    # Import lazily: predict.py invokes this module after defining its schema.
    from predict import view_features, map_to_percentile, confidence_adjusted_reaction, align_reaction_score
    cutoff = timestamp(event["knowledge_cutoff"])
    event_date = timestamp(event["event_datetime"])
    quarter = f"{event_date.year}Q{(event_date.month-1)//3+1}"
    previous = [h for h in history.get(ticker, []) if h["quarter"] < quarter
                and timestamp(h["event_datetime"]) < cutoff
                and timestamp(h["available_at"]) < cutoff][-3:]
    moves = [h["car1"] for h in previous]
    result, parts = {}, []
    for prefix, model in zip(("mini", "nano"), MODELS):
        view = views[model]
        values = view_features(view)
        values.update({
            "confidence": view.confidence, "positive_count": view.positive_surprises,
            "negative_count": view.negative_surprises,
            "raw_eps_surprise": view.eps_surprise_pct,
            "raw_revenue_surprise": view.revenue_surprise_pct,
            "raw_guidance_change": view.guidance_magnitude_pct,
            "revenue_known": int(view.revenue_vs_expectations != "unknown"),
            "eps_known": int(view.eps_vs_expectations != "unknown"),
            "guidance_known": int(view.guidance_action != "unknown"),
            "underlying_known": int(view.underlying_vs_expectations != "unknown"),
            "positioning_known": int(view.positioning != "unknown"),
            "base_score": map_to_percentile(view),
        })
        result.update({prefix+"_"+k: float(v) if v is not None else np.nan for k,v in values.items()})
        parts.append(confidence_adjusted_reaction(view))
    result.update({
        "history_n": float(len(previous)),
        "history_last_return": moves[-1] if moves else np.nan,
        "history_mean_return": float(np.mean(moves)) if moves else np.nan,
        "history_abs_return": float(np.mean(np.abs(moves))) if moves else np.nan,
        "history_beat_sell": sum(h["surprise"] is not None and h["surprise"] > 0 and h["car1"] < 0 for h in previous)/(len(previous)+2),
        "history_miss_relief": sum(h["surprise"] is not None and h["surprise"] < 0 and h["car1"] > 0 for h in previous)/(len(previous)+2),
    })
    stats = next((i.get("content") or {} for i in summary.get("items", [])
                  if i.get("id") == "option-implied-stats"), {})
    for source, name in (("implied_earnings_volatility", "option_vol"), ("skew_25_delta", "option_skew")):
        block = stats.get(source) or {}; value = block.get("value")
        result[name] = float(value) if block.get("status") == "ok" and isinstance(value, (int, float)) else np.nan
    price_values, last_close = features_from_frames(stock, market, cutoff)
    result.update({"price_"+k: v for k,v in price_values.items()})
    facts = next((i.get("content") for i in summary.get("items", [])
                  if i.get("id") == "earnings-call-facts"), [])
    text = " ".join(facts or []).lower()
    result.update({"text_"+name: float(bool(re.search(pattern, text))) for name,pattern in TOPICS.items()})
    baseline = align_reaction_score(float(np.mean(parts)))
    result["base_mean"] = baseline
    from predict import REACTION_SCALE_SLOPE
    result["base_disagreement"] = REACTION_SCALE_SLOPE * abs(parts[0]-parts[1])
    return result, baseline, last_close


def corrected_score(bundle: dict, features: dict, baseline: float) -> float:
    values = np.array([[features[name] for name in bundle["features"]]])
    slope = float(bundle["baseline_beta"][1])
    if not np.isfinite(slope) or abs(slope) < 1e-6:
        raise ValueError("Invalid correction scale")
    delta = float(bundle["model"].predict(values)[0]) * bundle["strength"] / slope
    score = float(np.clip(baseline + delta, 0, 1))
    if not np.isfinite(score):
        raise ValueError("Non-finite correction")
    return score


def adjust_forecast(baseline: float, *, event: dict, summary: dict,
                    ticker: str, views: dict) -> tuple[float, dict]:
    details = {"applied": False, "baseline": baseline}
    try:
        metadata = load_metadata()
        if not event.get("knowledge_cutoff") or not event.get("event_datetime"):
            return baseline, {**details, "reason": "missing_event_metadata"}
        if not available_for_event(metadata, event):
            return baseline, {**details, "reason": "information_after_event_cutoff"}
        if set(views) != set(MODELS) or os.environ.get("PROMPT_VERSION", "v2") != "v2" or os.environ.get("CONFIDENCE_SHRINK", "1") == "0":
            return baseline, {**details, "reason": "incompatible_model_views"}
        if event.get("prediction_deadline"):
            remaining = (timestamp(event["prediction_deadline"])-datetime.now(timezone.utc)).total_seconds()
            if remaining <= 2*PRICE_TIMEOUT_SECONDS + SUBMISSION_RESERVE_SECONDS:
                return baseline, {**details, "reason": "submission_deadline_near"}
        bundle, history = load_artifacts()
        with ThreadPoolExecutor(max_workers=2) as pool:
            stock_future = pool.submit(fetch_prices, ticker, event["knowledge_cutoff"])
            market_future = pool.submit(fetch_prices, "SPY", event["knowledge_cutoff"])
            stock, market = stock_future.result(), market_future.result()
        values, reconstructed, last_close = build_features(event, summary, ticker, views, history, stock, market)
        if abs(reconstructed-baseline) > 1e-10:
            raise ValueError("Baseline does not match the correction inputs")
        if not values["price_price_available"] or market is None:
            return baseline, {**details, "reason": "prices_unavailable"}
        score = corrected_score(bundle, values, baseline)
        return score, {**details, "applied": True, "corrected": score, "last_price_close": last_close,
                       "training_data_available_at": metadata["training_data_available_at"],
                       "selection_data_available_at": metadata["selection_data_available_at"]}
    except Exception as exc:
        return baseline, {**details, "reason": "correction_unavailable", "error_type": type(exc).__name__}
