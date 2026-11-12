"""Behavioral checks for data timing, fallback and the production integration."""
import numpy as np
import pandas as pd
import pytest
import httpx
from datetime import datetime, timedelta, timezone

import predict
from explaining_markets import statistical
from explaining_markets.market_features import chart_frame, features_from_frames


def test_selection_results_after_cutoff_block_even_an_older_trained_model():
    metadata = {"training_data_available_at": "2026-11-06T23:00:00Z",
                "selection_data_available_at": "2026-11-11T23:00:44Z"}
    assert not statistical.available_for_event(metadata, {"knowledge_cutoff": "2026-11-11T21:00:00Z"})
    assert statistical.available_for_event(metadata, {"knowledge_cutoff": "2026-11-12T21:00:00Z"})


def test_training_results_after_cutoff_also_block_correction():
    metadata = {"training_data_available_at": "2026-11-12T23:00:00Z",
                "selection_data_available_at": "2026-11-11T23:00:44Z"}
    assert not statistical.available_for_event(metadata, {"knowledge_cutoff": "2026-11-12T21:00:00Z"})


def test_ineligible_event_does_not_fetch_prices_or_load_model(monkeypatch):
    monkeypatch.setattr(statistical, "load_metadata", lambda: {
        "training_data_available_at": "2026-11-06T23:00:00Z",
        "selection_data_available_at": "2026-11-11T23:00:44Z"})
    def unexpected():
        raise AssertionError("Must not load an ineligible model")
    monkeypatch.setattr(statistical, "load_artifacts", unexpected)
    score, details = statistical.adjust_forecast(.61, event={
        "event_datetime": "2026-11-12T11:00:00Z", "knowledge_cutoff": "2026-11-11T21:00:00Z"},
        summary={}, ticker="TEST", views={})
    assert score == .61
    assert details["reason"] == "information_after_event_cutoff"


def test_incomplete_model_views_keep_existing_score(monkeypatch):
    monkeypatch.setattr(statistical, "load_metadata", lambda: {
        "training_data_available_at": "2026-11-06T23:00:00Z",
        "selection_data_available_at": "2026-11-11T23:00:44Z"})
    score, details = statistical.adjust_forecast(.61, event={
        "event_datetime": "2026-11-13T11:00:00Z", "knowledge_cutoff": "2026-11-12T21:00:00Z"},
        summary={}, ticker="TEST", views={"gpt-5-mini": object()})
    assert score == .61
    assert details["reason"] == "incompatible_model_views"


def test_market_data_failure_keeps_the_baseline(monkeypatch):
    monkeypatch.setattr(statistical, "load_metadata", lambda: {
        "training_data_available_at": "2026-11-06T23:00:00Z",
        "selection_data_available_at": "2026-11-11T23:00:44Z"})
    monkeypatch.setattr(statistical, "load_artifacts", lambda: ({}, {}))
    def fail(*args):
        raise httpx.ReadTimeout("Prices unavailable")
    monkeypatch.setattr(statistical, "fetch_prices", fail)
    score, details = statistical.adjust_forecast(.61, event={
        "event_datetime": "2026-11-13T11:00:00Z", "knowledge_cutoff": "2026-11-12T21:00:00Z"},
        summary={}, ticker="TEST", views={m: object() for m in statistical.MODELS})
    assert score == .61
    assert details["error_type"] == "ReadTimeout"


def test_near_deadline_skips_additional_work(monkeypatch):
    monkeypatch.setattr(statistical, "load_metadata", lambda: {
        "training_data_available_at": "2026-11-06T23:00:00Z",
        "selection_data_available_at": "2026-11-11T23:00:44Z"})
    score, details = statistical.adjust_forecast(.61, event={
        "event_datetime": "2026-11-13T11:00:00Z", "knowledge_cutoff": "2026-11-12T21:00:00Z",
        "prediction_deadline": (datetime.now(timezone.utc)+timedelta(seconds=10)).isoformat()},
        summary={}, ticker="TEST", views={m: object() for m in statistical.MODELS})
    assert score == .61
    assert details["reason"] == "submission_deadline_near"


def test_prices_after_cutoff_do_not_change_features():
    dates = pd.bdate_range("2026-08-03", periods=70, tz="America/New_York") + pd.Timedelta(hours=16)
    stock = pd.DataFrame({"close": np.arange(100.,170.), "volume": 1000.}, index=dates)
    before, last = features_from_frames(stock, stock, dates[-2])
    stock.loc[dates[-1], ["close", "volume"]] = [1000000., 1000000000.]
    after, after_last = features_from_frames(stock, stock, dates[-2])
    assert np.allclose(list(before.values()), list(after.values()), equal_nan=True)
    assert last == after_last == dates[-2].isoformat()


def test_intraday_timestamp_is_not_treated_as_a_completed_daily_bar():
    start = pd.Timestamp("2026-11-10T09:30:00", tz="America/New_York")
    data = {"meta": {"symbol": "TEST", "exchangeTimezoneName": "America/New_York"},
            "timestamp": [int(start.timestamp())],
            "indicators": {"adjclose": [{"adjclose": [100.]}], "quote": [{"volume": [1000]}]}}
    frame = chart_frame(data, "TEST")
    values, last = features_from_frames(frame, frame, "2026-11-10T15:00:00Z")
    assert values["price_available"] == 0
    assert last is None


def test_prediction_passes_both_named_views_and_event_to_correction(monkeypatch):
    from test_structured_strategy import view
    views = {"gpt-5-mini": view(.8, .9), "gpt-5.4-nano": view(.2, .6)}
    monkeypatch.setattr(predict, "ensemble_models", lambda: list(views))
    monkeypatch.setattr(predict, "_ask_structured", lambda **kw: views[kw["model"]])
    event = {"event_id": "TEST", "event_datetime": "2026-11-13T11:00:00Z",
             "knowledge_cutoff": "2026-11-12T21:00:00Z"}
    def correct(base, **kwargs):
        assert kwargs["views"] == views
        assert kwargs["event"] is event
        return .63, {"applied": True}
    monkeypatch.setattr(statistical, "adjust_forecast", correct)
    score, first = predict.predict_structured(summary={}, ticker="TEST", event_type="EARNINGS_RELEASE", event=event)
    assert score == .63
    assert first is views["gpt-5-mini"]
