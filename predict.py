"""★ THIS IS THE ONLY FILE YOU NEED TO EDIT. ★

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

The default implementation asks an OpenAI model for a calibrated percentile. If
`OPENAI_API_KEY` is not set, it returns a 0.5 baseline so the full deploy →
receive → submit round-trip still works without burning credits. Replace the body
of `predict` with whatever strategy you like — the only contract is the return
shape documented below.
"""

from __future__ import annotations

import json
import os

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

from explaining_markets.config import openai_model

_openai: OpenAI | None = None  # lazy: importing this file must not require a key
_openai_warned = False         # one-shot warning when no key is configured

# Timeouts, sized against the 5-minute prediction window that opens when your
# handler ACKs the webhook. Worst case is 15 + (120 x 2) + 15 = 270s, which
# fits with ~30s to spare. Nothing upstream retries a failed prediction — once
# the delivery is ACKed the platform considers it done — so the one retry here
# is the only one you get. Raising either value can push you past the deadline.
SUMMARY_TIMEOUT_SECONDS = 15.0
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_RETRIES = 1


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL to the event's materials
                                   (a JSON bundle of items; see `format_materials`)
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    summary = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    summary.raise_for_status()
    summary_json = summary.json()

    # One model call per focal asset, in series — so the LLM budget below is
    # per asset, not per event. Today every event carries a single asset; if
    # that changes and you need several, run them concurrently rather than
    # raising the timeout.
    strategy = os.environ.get("STRATEGY", "structured")
    out = []
    for asset in event["focal_assets"]:
        ticker = asset["identifier_value"]
        if strategy == "structured":
            p, view = predict_structured(summary=summary_json, ticker=ticker, event_type=event["event_type"])
            if view is not None:
                print(f"[INFO] {ticker}: p={p:.2f} guid={view.guidance_action} rev={view.revenue_vs_expectations} "
                      f"expected={view.expected_reaction:.2f} conf={view.confidence:.2f} | {view.rationale}")
        else:
            p = _ask_llm(summary=summary_json, ticker=ticker, event_type=event["event_type"])
        out.append({"identifier_value": ticker, "predicted_percentile": p})
    return out


# ----------------------------------------------------------------------
# Reading the event's materials.
#
# The document behind `information_url` is a bundle: a list of `items`, each
# with an `id`, a `kind`, and a `content` whose JSON type follows the kind.
# Today an earnings event can carry three:
#
#   earnings-call-facts   kind "facts"  a list of strings     always present
#   earnings-preview      kind "text"   one markdown string   may be absent
#   option-implied-stats  kind "stats"  an OBJECT of numbers  may be absent
#
# Select items by `id`, never by position, and expect kinds you have not seen:
# new ones can be added at any time.
# ----------------------------------------------------------------------

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"
OPTION_STATS_ID = "option-implied-stats"

# The preview is by far the largest item (often ~8,000 characters). It is the
# only one that gets cut, and it goes LAST in the prompt so a cut never costs
# you the facts or the option statistics.
PREVIEW_MAX_CHARS = 8000
OTHER_ITEM_MAX_CHARS = 2000


def _percent(block: object) -> str | None:
    """A `{value, status}` statistic as a percentage, or None if unavailable."""
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    value = block.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return f"{value * 100:+.1f}" if value < 0 else f"{value * 100:.1f}"


def _format_option_stats(stats: dict) -> str | None:
    volatility = _percent(stats.get("implied_earnings_volatility"))
    if volatility is None:
        return None
    lines = [
        f"- Implied earnings volatility: {volatility}% (the standard deviation of the "
        "stock's move on this release that option prices imply)",
    ]
    move = _percent(stats.get("implied_absolute_earnings_move"))
    if move is not None:
        lines.append(
            f"- Implied absolute move: {move}% (the size of move options price in; "
            "it says nothing about direction)"
        )
    skew = _percent(stats.get("skew_25_delta"))
    if skew is not None:
        lines.append(
            f"- 25-delta skew: {skew} volatility points (call minus put implied "
            "volatility; negative means downside protection is priced richer)"
        )
    else:
        lines.append("- 25-delta skew: unavailable (options too thinly traded)")
    as_of = stats.get("as_of")
    header = "Option-market expectations, measured before the release"
    if isinstance(as_of, str) and as_of:
        header += f" (as of {as_of})"
    return header + ":\n" + "\n".join(lines)


def format_materials(bundle: object) -> str:
    """Turn the event's bundle into the text the model reads.

    Each item gets its own labelled section. Items are found by `id`, so the
    order they arrive in does not matter, and an item this code has never heard
    of is included if it is text and skipped otherwise, never an error.
    """
    if not isinstance(bundle, dict):
        return ""
    raw_items = bundle.get("items")
    items = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []
    by_id = {i.get("id"): i for i in items}
    sections: list[str] = []

    facts = (by_id.get(FACTS_ID) or {}).get("content")
    if isinstance(facts, list) and facts:
        sections.append(
            "Facts from the earnings call:\n"
            + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        )

    stats = (by_id.get(OPTION_STATS_ID) or {}).get("content")
    if isinstance(stats, dict):
        formatted = _format_option_stats(stats)
        if formatted:
            sections.append(formatted)

    for item in items:
        if item.get("id") in (FACTS_ID, PREVIEW_ID, OPTION_STATS_ID):
            continue
        content = item.get("content")
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            content = "\n".join(content)
        if isinstance(content, str) and content.strip():
            label = item.get("id") or item.get("kind") or "item"
            sections.append(f"Additional material ({label}):\n{content[:OTHER_ITEM_MAX_CHARS]}")

    preview = (by_id.get(PREVIEW_ID) or {}).get("content")
    if isinstance(preview, str) and preview.strip():
        text = preview[:PREVIEW_MAX_CHARS]
        if len(preview) > PREVIEW_MAX_CHARS:
            text += "\n[preview truncated]"
        sections.append(
            "Research note written BEFORE the release (expectations, not results):\n" + text
        )

    return "\n\n".join(sections)


# ----------------------------------------------------------------------
# Structured strategy (default): extract a surprise vector, then map it.
#
# The LLM does not "give a feeling" about the release. It compares what the
# market expected going in (the pre-release preview: consensus, guidance,
# what is priced in) against what the company reported (the call facts), and
# fills a fixed set of fields. A linear map fitted on the historical archive
# turns the fields into a percentile; the model's own holistic judgement is
# blended in. Fallback: the single-call strategy below.
# ----------------------------------------------------------------------

from typing import Literal

PREVIEW_MAX_CHARS_STRUCTURED = 14000


class SurpriseView(BaseModel):
    """What the release delivered relative to what the market expected."""

    revenue_vs_expectations: Literal["beat", "inline", "miss", "unknown"]
    revenue_surprise_pct: float | None = Field(
        description="Reported revenue vs consensus, in percent (e.g. 2.5 for a 2.5% beat). null if unknown."
    )
    eps_vs_expectations: Literal["beat", "inline", "miss", "unknown"]
    eps_surprise_pct: float | None = Field(description="Reported EPS vs consensus, in percent. null if unknown.")
    guidance_action: Literal["raised", "maintained", "lowered", "withdrawn", "initiated", "none", "unknown"]
    guidance_vs_consensus: Literal["above", "inline", "below", "unknown"] = Field(
        description="New or reiterated guidance midpoint relative to consensus for that period."
    )
    guidance_magnitude_pct: float | None = Field(
        description="Size of the guidance change at the midpoint, in percent; negative for cuts. null if none."
    )
    margin_trend: Literal["expanding", "stable", "contracting", "unknown"]
    demand_tone: Literal["strong", "neutral", "weak"] = Field(
        description="Management's description of demand / orders / pipeline."
    )
    positive_surprises: int = Field(ge=0, le=10, description="Count of notable positives NOT already expected.")
    negative_surprises: int = Field(ge=0, le=10, description="Count of notable negatives NOT already expected.")
    priced_in: float = Field(
        ge=0.0, le=1.0,
        description="How much of the reported outcome was already anticipated before the release (1 = fully).",
    )
    expected_reaction: float = Field(
        ge=0.0, le=1.0,
        description="Your overall percentile for the next-day abnormal return across the quarter's events.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in expected_reaction.")
    rationale: str = Field(description="One or two sentences: the key driver of the reaction.")
    # v2 fields: the bar is the price, not the consensus.
    one_time_items_drive_beat: bool = Field(
        default=False,
        description="True if the beat or the guidance raise is mainly due to non-recurring items (tariff refunds, tax benefits, asset sales, one-off contracts).",
    )
    underlying_vs_expectations: Literal["beat", "inline", "miss", "unknown"] = Field(
        default="unknown", description="Result vs expectations EXCLUDING one-time items."
    )
    growth_trajectory: Literal["accelerating", "stable", "decelerating", "unknown"] = Field(
        default="unknown", description="Direction of growth implied by the quarter and the next-period guide."
    )
    positioning: Literal["priced_for_perfection", "neutral", "beaten_down", "unknown"] = Field(
        default="unknown",
        description="How the stock went in, from the note: big run-up / high multiple / bullish narrative = priced_for_perfection; sharp decline, known bad news, bearish narrative = beaten_down.",
    )
    vs_bear_case: Literal["better", "inline", "worse", "unknown"] = Field(
        default="unknown",
        description="Compared with what a pessimist expected going in (the bear case in the note, pre-announced bad news), did the print come in better, inline or worse?",
    )


STRUCTURED_SYSTEM_PROMPT = """\
You are a sell-side equity analyst scoring an earnings release the moment it lands.

You receive (a) facts from the earnings call and, when available, (b) a research
note written BEFORE the release with consensus numbers, company guidance and what
the market already expects, and (c) option-market expectations.

Your job is to compare WHAT WAS REPORTED against WHAT WAS EXPECTED and fill every
field. Rules:
- Use the pre-release note as the yardstick. A number that merely matches guidance
  or consensus is "inline", not a beat. "Record" or "growth" language is not a
  beat unless it exceeds what was expected.
- Guidance matters more than the reported quarter. Classify the guidance action
  and compare the new midpoint to consensus if the note gives one.
- Count only surprises that were NOT already anticipated in the note.
- If no pre-release note is provided, use the facts' own comparisons to
  expectations (e.g. "exceeding consensus"), and mark fields "unknown" when the
  facts do not say.
- expected_reaction is a cross-sectional percentile across all earnings events
  this quarter: 0 = most negative reaction, 0.5 = median, 1 = most positive.
  Base rates: about 25% of events land above 0.75 and 25% below 0.25. Keep
  mixed or fully-anticipated outcomes near 0.40-0.60.
"""

STRUCTURED_SYSTEM_PROMPT_V2 = STRUCTURED_SYSTEM_PROMPT + """
The bar the stock has to clear is set by its PRICE, not by the sell-side consensus.
Apply these rules, in this order, before setting expected_reaction:

1. One-time items do not count. A beat or a guidance raise that comes from
   tariff refunds, tax benefits, asset sales, insurance recoveries or other
   non-recurring items is at best "inline". Score the UNDERLYING result; if the
   underlying margin or growth deteriorated, the reaction is negative even though
   the headline numbers look great.
2. Priced for perfection. If the note shows the stock ran up hard into the print,
   trades on a high multiple, or carries a bullish narrative, then a modest
   beat-and-raise is merely expected (0.40-0.55). Any sign of deceleration — a
   softer next-quarter guide, slowing key metrics, delayed expansion, a trimmed
   outlook anywhere — is a strong negative for such a stock (0.10-0.30), no
   matter how good the reported quarter was.
3. Relief. If the bad news was pre-announced, is already in the note's bear case,
   or the stock is beaten down, then a print that is NOT WORSE than that bear
   case is neutral-to-positive (0.50-0.70), and anything genuinely better than
   feared is strongly positive (0.70-0.90). Do not score known bad news twice.
4. Only when none of the above dominates, fall back to the plain comparison of
   reported vs consensus, with guidance weighing more than the quarter.

Use the full confidence range: 0.85+ only when the print is unambiguous in one
direction AND the positioning does not fight it; 0.3-0.5 when the signals
conflict (e.g. great quarter but decelerating guide on a hot stock).
"""


def structured_prompt() -> str:
    return STRUCTURED_SYSTEM_PROMPT_V2 if os.environ.get("PROMPT_VERSION", "v2") == "v2" else STRUCTURED_SYSTEM_PROMPT


def _ask_structured(
    *, summary: dict, ticker: str, event_type: str, model: str | None = None
) -> SurpriseView | None:
    """One structured-output call. Returns None if no key is configured or the model refuses."""
    global _openai
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    if _openai is None:
        _openai = OpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)

    materials = format_materials_structured(summary)
    if not materials:
        materials = json.dumps(summary)[:PREVIEW_MAX_CHARS_STRUCTURED]

    user_prompt = (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"{materials}\n\n"
        f"Compare reported vs expected for {ticker} and fill every field."
    )
    resp = _openai.chat.completions.parse(
        model=model or openai_model(),
        messages=[
            {"role": "system", "content": structured_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        response_format=SurpriseView,
    )
    return resp.choices[0].message.parsed


def format_materials_structured(bundle: object) -> str:
    """Like format_materials, but with the preview FIRST (it is the yardstick) and a longer cap."""
    if not isinstance(bundle, dict):
        return ""
    raw_items = bundle.get("items")
    items = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []
    by_id = {i.get("id"): i for i in items}
    sections: list[str] = []

    preview = (by_id.get(PREVIEW_ID) or {}).get("content")
    if isinstance(preview, str) and preview.strip():
        text = preview[:PREVIEW_MAX_CHARS_STRUCTURED]
        if len(preview) > PREVIEW_MAX_CHARS_STRUCTURED:
            text += "\n[preview truncated]"
        sections.append("=== (b) PRE-RELEASE RESEARCH NOTE: what the market expected ===\n" + text)
    else:
        sections.append("=== (b) PRE-RELEASE RESEARCH NOTE === not available for this event.")

    facts = (by_id.get(FACTS_ID) or {}).get("content")
    if isinstance(facts, list) and facts:
        sections.append(
            "=== (a) FACTS FROM THE EARNINGS CALL: what was reported ===\n"
            + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        )

    stats = (by_id.get(OPTION_STATS_ID) or {}).get("content")
    if isinstance(stats, dict):
        formatted = _format_option_stats(stats)
        if formatted:
            sections.append("=== (c) " + formatted)

    for item in items:
        if item.get("id") in (FACTS_ID, PREVIEW_ID, OPTION_STATS_ID):
            continue
        content = item.get("content")
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            content = "\n".join(content)
        if isinstance(content, str) and content.strip():
            label = item.get("id") or item.get("kind") or "item"
            sections.append(f"Additional material ({label}):\n{content[:OTHER_ITEM_MAX_CHARS]}")
    return "\n\n".join(sections)


# Numeric encoding of the categorical fields.
_SIGN = {
    "beat": 1.0, "inline": 0.0, "miss": -1.0, "unknown": 0.0,
    "raised": 1.0, "maintained": 0.0, "lowered": -1.0, "withdrawn": -1.0, "initiated": 0.0, "none": 0.0,
    "above": 1.0, "below": -1.0,
    "expanding": 1.0, "stable": 0.0, "contracting": -1.0,
    "strong": 1.0, "neutral": 0.0, "weak": -1.0,
    "accelerating": 1.0, "decelerating": -1.0,
    "priced_for_perfection": -1.0, "beaten_down": 1.0,
    "better": 1.0, "worse": -1.0,
}


def _clip(v: float | None, lo: float, hi: float) -> float:
    if v is None:
        return 0.0
    return max(lo, min(hi, float(v)))


def view_features(view: SurpriseView) -> dict[str, float]:
    """The regressors the linear map consumes. Keep names stable: the weights below depend on them."""
    net = float(view.positive_surprises - view.negative_surprises)
    return {
        "rev": _SIGN[view.revenue_vs_expectations],
        "rev_pct": _clip(view.revenue_surprise_pct, -20.0, 20.0) / 10.0,
        "eps": _SIGN[view.eps_vs_expectations],
        "guid": _SIGN[view.guidance_action],
        "guid_cons": _SIGN[view.guidance_vs_consensus],
        "guid_pct": _clip(view.guidance_magnitude_pct, -30.0, 30.0) / 10.0,
        "margin": _SIGN[view.margin_trend],
        "tone": _SIGN[view.demand_tone],
        "net_surprises": max(-5.0, min(5.0, net)),
        "priced_in": view.priced_in - 0.5,
        "expected": view.expected_reaction - 0.5,
        "expected_x_conf": (view.expected_reaction - 0.5) * view.confidence,
        "one_time": -1.0 if view.one_time_items_drive_beat else 0.0,
        "underlying": _SIGN[view.underlying_vs_expectations],
        "trajectory": _SIGN[view.growth_trajectory],
        "positioning": _SIGN[view.positioning],
        "vs_bear": _SIGN[view.vs_bear_case],
    }


# Fitted on the historical archive (see backtest/fit_mapping.py). Intercept is 0.5
# by construction; the map is a percentile-shaped linear score. Re-fit when the
# prompt or the field set changes.
CONFIDENCE_SHRINK = os.environ.get("CONFIDENCE_SHRINK", "1") != "0"

MAPPING_WEIGHTS: dict[str, float] = {
    "rev": 0.03,
    "rev_pct": 0.01,
    "eps": 0.02,
    "guid": 0.06,
    "guid_cons": 0.04,
    "guid_pct": 0.01,
    "margin": 0.015,
    "tone": 0.02,
    "net_surprises": 0.01,
    "priced_in": -0.02,
    "expected": 0.25,
    "expected_x_conf": 0.15,
    "one_time": 0.03,
    "underlying": 0.02,
    "trajectory": 0.03,
    "positioning": 0.02,
    "vs_bear": 0.03,
}


def map_to_percentile(view: SurpriseView, weights: dict[str, float] | None = None) -> float:
    """Linear score, then shrink toward the median by the model's confidence.

    The archive shows the model's confidence is informative (high-confidence
    predictions track the realized ranking three times better than low-confidence
    ones), so a low-confidence view should sit closer to 0.5 in the cross-section.
    """
    w = weights or MAPPING_WEIGHTS
    feats = view_features(view)
    score = 0.5 + sum(w.get(k, 0.0) * v for k, v in feats.items())
    if CONFIDENCE_SHRINK:
        score = 0.5 + (score - 0.5) * max(0.2, min(1.0, view.confidence))
    return max(0.0, min(1.0, score))


def ensemble_models() -> list[str]:
    """Models whose structured views are averaged. One entry = no ensemble.

    Backtest (Q3 2026 contest window): gpt-5-mini alone 0.167, gpt-5.4-nano alone
    0.156, their average 0.176. Averaging the same prompt across two models only
    helped once the v2 rules were in place.
    """
    raw = os.environ.get("ENSEMBLE_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or [openai_model()]


def predict_structured(*, summary: dict, ticker: str, event_type: str) -> tuple[float, SurpriseView | None]:
    """Structured strategy: one view per ensemble model, scores averaged; fallback to the single-call strategy."""
    from concurrent.futures import ThreadPoolExecutor

    models = ensemble_models()

    def one(model: str) -> SurpriseView | None:
        try:
            return _ask_structured(summary=summary, ticker=ticker, event_type=event_type, model=model)
        except Exception as exc:  # noqa: BLE001 - never let one model kill the prediction
            print(f"[WARN] structured call failed for {model}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        views = [v for v in ex.map(one, models) if v is not None]
    if not views:
        return _ask_llm(summary=summary, ticker=ticker, event_type=event_type), None
    score = sum(map_to_percentile(v) for v in views) / len(views)
    return score, views[0]


# ----------------------------------------------------------------------
# Default strategy: a single calibrated LLM call per asset.
# Swap this out, or rewrite `predict` entirely, to enter your own model.
# ----------------------------------------------------------------------


class Prediction(BaseModel):
    """Structured response shape for the LLM call.

    The `Field(ge=0, le=1)` constraint flows through into the JSON schema OpenAI's
    structured-outputs mode enforces during decoding, so the model is guaranteed to
    return a percentile in [0, 1] — no manual clamping or fallback parsing needed.
    """

    predicted_percentile: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """\
You are a senior equity analyst predicting how a stock will react to an event.

Predict a single percentile in [0, 1] for how the focal asset's next-day
abnormal return will rank across all of the quarter's event outcomes:
0 = the quarter's most negative reaction, 0.50 = median, 1 = its most positive.
The relevant return is the *unexpected*, market-adjusted return — a
great-but-fully-priced-in beat is not a top-decile event.

Calibration discipline:
- Long-run base rates: about 25% of events land "up" (>0.75), 50% "neutral"
  (0.25-0.75), 25% "down" (<0.25). Default toward 0.40-0.60 when signals are
  mixed or modest.
- Reserve values above 0.80 or below 0.20 for cases with unambiguous,
  multi-signal evidence. Do not exceed 0.90 or fall below 0.10 without
  overwhelming, lopsided evidence.
- Tone alone (confident vs hedging language) should move you no more than
  ~0.03 absent quantitative confirmation.
"""


def _ask_llm(*, summary: dict, ticker: str, event_type: str) -> float:
    """Ask the configured model for a calibrated percentile via structured outputs.

    Returns the model's `predicted_percentile`. Falls back to 0.5 if no
    `OPENAI_API_KEY` is configured or the model refuses; the [0, 1] bound is
    enforced by the JSON schema, not by us.
    """
    global _openai, _openai_warned
    if not os.environ.get("OPENAI_API_KEY"):
        if not _openai_warned:
            print(
                "[WARN] OPENAI_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key (or edit predict.py) for real predictions."
            )
            _openai_warned = True
        return 0.5
    if _openai is None:
        # picks up OPENAI_API_KEY from env
        _openai = OpenAI(
            timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
        )

    materials = format_materials(summary)
    if not materials:
        # Not a bundle this code recognises. Show the model what arrived rather
        # than nothing.
        materials = json.dumps(summary)[:PREVIEW_MAX_CHARS]

    user_prompt = (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"Event materials:\n{materials}\n\n"
        "Weigh, in roughly this order:\n"
        "  1. Quantitative surprise vs expectations — revenue, EPS, segment metrics.\n"
        "  2. Guidance / outlook — raises, holds, cuts vs the prior trajectory.\n"
        "  3. Strategic shifts — product launches, M&A, capital allocation, leadership.\n"
        "  4. Tone and confidence in management commentary (small weight).\n"
        "  5. Risks called out — regulatory, supply chain, demand, competition.\n\n"
        f"Predict the next-day unexpected-return percentile for {ticker}."
    )

    resp = _openai.chat.completions.parse(
        model=openai_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Prediction,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        return 0.5  # model refused; competition expects a number
    return parsed.predicted_percentile
