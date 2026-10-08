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
the LLM for events not yet cached. Cached runs for the current strategy are
committed so you can score variants of the mapping without spending anything.

## Reference numbers (Q3 2026 contest window, dR2 on the common sample)

| run | dR2 |
|---|---|
| original starter, gpt-5.4-nano | 0.096 |
| structured v1, nano | 0.138 |
| structured v2 + confidence shrink, nano | 0.156 |
| structured v2, gpt-5-mini | 0.167 |
| v2 mini + nano averaged (deployed) | 0.176 |
| official baseline GLM | 0.124 |
| last quarter's winner | 0.168 |
