"""Daily-price features restricted to completed sessions before the cutoff."""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")


def chart_frame(data: dict, symbol: str):
    meta = data.get("meta") or {}
    if meta.get("exchangeTimezoneName") != "America/New_York":
        return None
    if str(meta.get("symbol", "")).casefold() != symbol.casefold():
        return None
    indicators = data.get("indicators") or {}
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    quote = (indicators.get("quote") or [{}])[0]
    timestamps = data.get("timestamp") or []
    if adjusted is None or len(adjusted) != len(timestamps):
        return None
    volumes = quote.get("volume") or [None] * len(timestamps)
    rows = []
    for timestamp, close, volume in zip(timestamps, adjusted, volumes):
        if close is None or not np.isfinite(close) or close <= 0:
            continue
        date = datetime.fromtimestamp(timestamp, NY).date()
        # An early-close session is conservatively unavailable until 16:00 NY.
        available = datetime.combine(date, time(16), tzinfo=NY)
        rows.append({"available": available, "close": float(close),
                     "volume": float(volume) if volume is not None else np.nan})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates("available").set_index("available").sort_index()


def features_from_frames(stock, market, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff must include a timezone")
    result = {"price_available": 0.}
    for n in [5, 20, 60]:
        result[f"return_{n}"] = np.nan
        result[f"relative_spy_{n}"] = np.nan
    for name in ["vol_20", "vol_60", "vol_ratio", "drawdown_60", "drawdown_252",
                 "range_position_60", "volume_ratio_5_60", "last_price_age_days"]:
        result[name] = np.nan
    if stock is None:
        return result, None
    past = stock.loc[stock.index <= cutoff]
    if past.empty:
        return result, None
    prices = past["close"].to_numpy()
    result["price_available"] = 1.
    result["last_price_age_days"] = (cutoff - past.index[-1]).total_seconds() / 86400
    market_past = market.loc[market.index <= cutoff] if market is not None else None
    for n in [5, 20, 60]:
        if len(prices) <= n:
            continue
        value = prices[-1] / prices[-n-1] - 1
        result[f"return_{n}"] = float(value)
        if market_past is not None:
            first = market_past.loc[market_past.index <= past.index[-n-1], "close"]
            last = market_past.loc[market_past.index <= past.index[-1], "close"]
            if not first.empty and not last.empty:
                result[f"relative_spy_{n}"] = float(value - (last.iloc[-1] / first.iloc[-1] - 1))
    for n in [20, 60]:
        if len(prices) > n:
            result[f"vol_{n}"] = float(np.std(np.diff(np.log(prices[-n-1:])), ddof=1) * np.sqrt(252))
    if result["vol_60"] > 0:
        result["vol_ratio"] = result["vol_20"] / result["vol_60"]
    for n in [60, 252]:
        if len(prices) >= n:
            result[f"drawdown_{n}"] = float(prices[-1] / max(prices[-n:]) - 1)
    if len(prices) >= 60:
        low, high = min(prices[-60:]), max(prices[-60:])
        result["range_position_60"] = float((prices[-1] - low) / (high - low)) if high > low else .5
        volumes = past["volume"].to_numpy()
        reference = np.nanmean(volumes[-60:])
        if reference > 0:
            result["volume_ratio_5_60"] = float(np.nanmean(volumes[-5:]) / reference)
    return result, past.index[-1].isoformat()
