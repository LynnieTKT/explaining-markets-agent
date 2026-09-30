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
    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ask_llm(
                summary=summary_json,
                ticker=asset["identifier_value"],
                event_type=event["event_type"],
            ),
        }
        for asset in event["focal_assets"]
    ]


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
