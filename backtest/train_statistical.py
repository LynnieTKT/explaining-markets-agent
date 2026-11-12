"""Train a fixed correction artifact from an explicitly supplied data manifest.

Run with --manifest PATH. This is an offline command, never automatic retraining
inside the webhook. The manifest names frozen archives, model views and prices.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
from predict import SurpriseView
from explaining_markets.market_features import chart_frame
from explaining_markets.statistical import build_features, timestamp, MODELS


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(values):
    values = np.array(values, dtype=float)
    ranks = (pd.Series(values).rank(method="average").to_numpy()-1)/(len(values)-1)
    order = np.argsort(values)
    return values[order], ranks[order]


def train(config):
    source_records, quarters = {}, {}
    source_hashes = {}
    for file in config["archives"]:
        path = Path(file)
        quarter = path.name.split("_")[-1].removesuffix(".jsonl.gz")
        with gzip.open(path, "rt") as source:
            rows = [json.loads(line) for line in source]
        quarters[quarter] = rows
        source_records.update({r["event_id"]: (quarter, r) for r in rows})
        source_hashes["archive_"+quarter] = file_hash(path)
    history = defaultdict(list)
    for quarter, row in source_records.values():
        for asset in row.get("focal_assets", []):
            symbol = asset["identifier_value"]
            value = (row.get("event_returns") or {}).get(symbol) or {}
            if value.get("car1") is not None and row.get("returns_computed_at"):
                history[symbol].append({"quarter": quarter, "event_datetime": row["event_datetime"],
                    "available_at": row["returns_computed_at"], "car1": float(value["car1"]),
                    "surprise": (row.get("metrics") or {}).get("earnings_surprise", {}).get("surprise")})
    for rows in history.values():
        rows.sort(key=lambda r: timestamp(r["event_datetime"]))
    views, prior_ids = {}, set()
    for model, file in config["prior_views"].items():
        source_hashes["prior_"+model] = file_hash(file)
        for line in Path(file).read_text().splitlines():
            row = json.loads(line)
            if row.get("view"):
                views.setdefault(row["event_id"], {})[model] = SurpriseView(**row["view"])
                prior_ids.add(row["event_id"])
    source_hashes["current_views"] = file_hash(config["views"])
    for line in Path(config["views"]).read_text().splitlines():
        row = json.loads(line)
        if row.get("status", "ok") == "ok" and row.get("view"):
            views.setdefault(row["event_id"], {})[row["model_requested"]] = SurpriseView(**row["view"])
    car_values, surprise_values = [], []
    for row in quarters["2026Q3"]:
        es = (row.get("metrics") or {}).get("earnings_surprise") or {}
        for asset in row.get("focal_assets", []):
            leg = (row.get("event_returns") or {}).get(asset["identifier_value"]) or {}
            if leg.get("car1") is None:
                continue
            car_values.append(float(leg["car1"]))
            if es.get("surprise_status") == "ok" and es.get("surprise") is not None:
                surprise_values.append(float(es["surprise"]))
    car_curve, surprise_curve = curve(car_values), curve(surprise_values)
    price_manifest_path = Path(config["price_manifest"])
    price_manifest = json.loads(price_manifest_path.read_text())
    source_hashes["prices_manifest"] = file_hash(price_manifest_path)
    price_cache, price_hashes = {}, {}

    def prices(symbol):
        if symbol in price_cache:
            return price_cache[symbol]
        value = price_manifest["symbols"].get(symbol)
        path = price_manifest_path.parent / value if value else None
        if path is None or not path.exists():
            price_cache[symbol] = None
        else:
            price_hashes[symbol] = file_hash(path)
            with gzip.open(path, "rt") as source:
                price_cache[symbol] = chart_frame(json.load(source)["data"], symbol)
        return price_cache[symbol]

    feature_names = config["feature_order"]
    xs, ys, baselines, surprises, weights, training_ids = [], [], [], [], [], []
    as_of = timestamp(config["as_of"])
    for eid, pair in views.items():
        if set(pair) != set(MODELS) or eid not in source_records:
            continue
        quarter, row = source_records[eid]
        if not row.get("returns_computed_at") or timestamp(row["returns_computed_at"]) > as_of:
            continue
        es = (row.get("metrics") or {}).get("earnings_surprise") or {}
        if es.get("surprise_status") != "ok" or es.get("surprise") is None:
            continue
        symbol = row["focal_assets"][0]["identifier_value"]
        x, baseline, _ = build_features(row, row["disclosure"], symbol, pair, history, prices(symbol), prices("SPY"))
        if set(x) != set(feature_names):
            raise ValueError("Training features differ from the validated feature schema")
        xs.append([x[k] for k in feature_names])
        baselines.append(baseline)
        ys.append(float(np.interp(row["event_returns"][symbol]["car1"], *car_curve)))
        surprises.append(float(np.interp(es["surprise"], *surprise_curve)))
        weights.append(.25 if eid in prior_ids else 1.)
        training_ids.append(eid)
    x, y, w = np.array(xs), np.array(ys), np.array(weights)
    design = np.c_[np.ones(len(y)), baselines, surprises]
    beta = np.linalg.lstsq(design*w[:,None]**.5, y*w**.5, rcond=None)[0]
    if abs(beta[1]) < 1e-6:
        raise ValueError("Unable to estimate the baseline scale")
    model = HistGradientBoostingRegressor(max_iter=80, learning_rate=.05, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=20, early_stopping=False, random_state=20261111)
    model.fit(x, y-design@beta, sample_weight=w)
    bundle = {"model": model, "baseline_beta": beta, "features": feature_names, "strength": .5}
    destination = Path(config["output_dir"]); destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "correction.joblib")
    with gzip.open(destination / "history.json.gz", "wt") as output:
        json.dump(history, output)
    source_hashes["price_files"] = hashlib.sha256(json.dumps(price_hashes,sort_keys=True).encode()).hexdigest()
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_available_at": max((source_records[e][1]["returns_computed_at"] for e in training_ids), key=timestamp),
        "selection_data_available_at": config["selection_data_available_at"],
        "model_sha256": file_hash(destination / "correction.joblib"),
        "history_sha256": file_hash(destination / "history.json.gz"),
        "models": list(MODELS), "prompt_version": "v2", "preview_limit": 24000,
        "feature_count": len(feature_names), "training_events": len(training_ids),
        "training_ids": training_ids, "source_hashes": source_hashes,
        "libraries": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "parameters": {"max_iter": 80, "learning_rate": .05, "max_leaf_nodes": 15,
            "min_samples_leaf": 40, "l2_regularization": 20, "prior_weight": .25, "strength": .5},
    }
    timestamp(metadata["selection_data_available_at"])
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = train(json.loads(Path(args.manifest).read_text()))
    print(json.dumps({k:v for k,v in result.items() if k != "training_ids"}, indent=2))
