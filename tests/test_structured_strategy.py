"""Regression checks for confidence-adjusted overall judgments and ensembling."""
import pytest
import predict as strategy


def view(expected, confidence):
    return strategy.SurpriseView(
        revenue_vs_expectations="beat", revenue_surprise_pct=5.0,
        eps_vs_expectations="beat", eps_surprise_pct=10.0,
        guidance_action="raised", guidance_vs_consensus="above", guidance_magnitude_pct=5.0,
        margin_trend="expanding", demand_tone="strong",
        positive_surprises=5, negative_surprises=0, priced_in=0.9,
        expected_reaction=expected, confidence=confidence,
        rationale="Headline beats were anticipated; the overall judgment also accounts for forward risks.",
    )


@pytest.mark.parametrize("expected,confidence,result", [
    (0.8, 0.5, 0.65),
    (0.2, 0.75, 0.275),
    (1.0, 0.0, 0.5),
    (0.0, 0.0, 0.5),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
])
def test_confidence_adjusts_overall_judgment_without_readding_headline_beats(monkeypatch, expected, confidence, result):
    monkeypatch.setattr(strategy, "CONFIDENCE_SHRINK", True)
    assert strategy.confidence_adjusted_reaction(view(expected, confidence)) == pytest.approx(result)


def test_existing_confidence_switch_is_respected(monkeypatch):
    monkeypatch.setattr(strategy, "CONFIDENCE_SHRINK", False)
    assert strategy.confidence_adjusted_reaction(view(0.8, 0.1)) == 0.8


def test_models_are_adjusted_individually_before_averaging(monkeypatch):
    monkeypatch.setattr(strategy, "CONFIDENCE_SHRINK", True)
    monkeypatch.setattr(strategy, "ensemble_models", lambda: ["first", "second"])
    views = {"first": view(0.8, 0.9), "second": view(0.2, 0.6)}
    monkeypatch.setattr(strategy, "_ask_structured", lambda **kw: views[kw["model"]])
    score, first = strategy.predict_structured(summary={}, ticker="TEST", event_type="EARNINGS_RELEASE")
    assert score == pytest.approx((0.77 + 0.32) / 2)
    assert first is views["first"]


def test_failed_model_does_not_dilute_successful_model(monkeypatch):
    monkeypatch.setattr(strategy, "CONFIDENCE_SHRINK", True)
    monkeypatch.setattr(strategy, "ensemble_models", lambda: ["failed", "available"])
    successful = view(0.8, 0.9)

    def call(**kw):
        if kw["model"] == "failed":
            raise RuntimeError("model unavailable")
        return successful

    monkeypatch.setattr(strategy, "_ask_structured", call)
    score, first = strategy.predict_structured(summary={}, ticker="TEST", event_type="EARNINGS_RELEASE")
    assert score == pytest.approx(0.77)
    assert first is successful


def test_no_structured_views_uses_existing_direct_fallback(monkeypatch):
    monkeypatch.setattr(strategy, "ensemble_models", lambda: ["first", "second"])
    monkeypatch.setattr(strategy, "_ask_structured", lambda **kw: None)
    monkeypatch.setattr(strategy, "_ask_llm", lambda **kw: 0.61)
    score, first = strategy.predict_structured(summary={}, ticker="TEST", event_type="EARNINGS_RELEASE")
    assert score == 0.61
    assert first is None
