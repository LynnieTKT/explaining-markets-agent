# Backtest harness

Replays `predict.py` against the historical archive and scores it with the
competition's own scorer (ported in the official `examples` repo), so numbers
here match the public leaderboard float for float.

## Setup

```bash
# once: the official examples repo next to this one, for the scorer + archive
git clone https://github.com/explaining-markets/examples.git ../../examples
(cd ../../examples && uv sync && cp .env.example .env)   # paste EM_API_KEY, set EM_API_BASE_URL to prod
(cd ../../examples && uv run python scripts/download_archive.py --event-type EARNINGS_RELEASE)

cd backtest && uv sync
```

`run_backtest.py` reads `../.env` (the deploy `.env`) for `OPENAI_API_KEY`.

## Run

```bash
export PYTHONPATH=../../examples/src:..:../src

# contest window of Q3 2026 (578 scorable events), structured strategy, v2 prompt
OPENAI_MODEL=gpt-5-mini PROMPT_VERSION=v2 uv run python run_backtest.py \
    --strategy structured --label my-change

uv run python analyze_archive.py      # what explains returns beyond the surprise (no LLM calls)
uv run python fit_mapping.py --label structured-v2-nano-q3   # fit/ablate the linear map
```

Model outputs are cached per label in `cache/<label>.jsonl`; re-runs only call
the LLM for events not yet cached. Cached structured outputs are committed so
you can recalculate their scores with alternative mappings without new model calls.

## Reference numbers (Q3 2026 contest window, dR2 on the common sample)

The structured-strategy reference results below use the original feature map,
retained in `map_to_percentile` for replay comparisons. The active strategy
averages confidence-adjusted overall reaction estimates and aligns their scale
with the prior submission using paired Q3 forecasts.
Eligible events can also use the bundled historical statistical correction;
the table below predates that component.

| run | dR2 |
|---|---|
| original starter, gpt-5.4-nano | 0.096 |
| structured v1, nano | 0.138 |
| structured v2 + confidence shrink, nano | 0.156 |
| structured v2, gpt-5-mini | 0.167 |
| v2 mini + nano averaged (original feature map) | 0.176 |
| official baseline GLM | 0.124 |
| last quarter's winner | 0.168 |

## Training the statistical correction

Run from the repository root:

```bash
uv run python backtest/train_statistical.py --manifest /path/to/training-inputs.json
```

The input manifest contains:

| Field | Required content |
|---|---|
| `archives` | Paths to frozen quarterly archive files, including Q3 2026 and the latest target-quarter snapshot |
| `prior_views` | Model name to Q3 structured-response JSONL path, for `gpt-5-mini` and `gpt-5.4-nano` |
| `views` | Current-quarter JSONL with `event_id`, `model_requested`, and `view` for each successful response |
| `price_manifest` | JSON file whose `symbols` map identifies the saved Yahoo daily-chart records |
| `feature_order` | The validated ordered feature-name list; keep the order fixed across training and inference |
| `as_of` | Timezone-aware latest permissible training-data availability time |
| `selection_data_available_at` | Latest availability time of the results used to select this method, including evaluation data |
| `output_dir` | Artifact directory, normally `src/explaining_markets/stat_model` |

The command writes a fixed model, company-history data and metadata containing
source hashes, training event IDs, library versions and both information-time
boundaries. Model input files contain model judgments, not current-event realized
returns. True historical outcomes are used only for training and evaluation.

Before deployment, compare the production features and predictions with the
validated replay, verify artifact inclusion in the package, and check market-data
access in the deployment environment. The runtime uses the base forecast when an
event's cutoff predates either recorded information boundary. Update the public
Model Card before activating a materially changed method.
