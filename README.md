# OECD GDP Forecasting: LSTM vs. Transformer Under Data Scarcity

Comparing an LSTM and a causal Transformer on next-quarter GDP growth forecasting
across a 29-country OECD panel, with a focus on how each architecture behaves
under short, noisy, data-scarce sequences rather than just which one wins.

## Summary

GDP forecasting from a single country's history (e.g., US-only FRED data) gives
only ~113 quarters of data, far too little to train either architecture
meaningfully. This project instead builds an 8-quarter-lookback, cross-country
panel from the OECD SDMX API (29 countries, 3,924 country-quarters, 10 features
per quarter), and trains both models with a shared training loop so any
performance difference is attributable to architecture, not implementation.

Both models embed country identity into a learned vector and are evaluated against a naive per-country
mean-growth baseline. 


## Repo structure

| File | Purpose |
|---|---|
| `data.py` | Fetches and assembles the OECD panel from the SDMX API into `oecd_panel.csv` |
| `preprocessing.py` | Loads the panel, builds 8-quarter sliding windows, splits into train/val/test by date |
| `lstm.py` | LSTM model with learned country embedding |
| `transformer.py` | Causal (masked) Transformer model with learned country + positional embeddings |
| `training.py` | Shared training loop for both architectures, multi-seed runs, early stopping, naive-baseline comparison |
| `country_failure_analysis.py` | Cross-references per-country baseline-beating results against GDP volatility and training history length |
| `oecd_panel.csv` | The assembled dataset (see note below) |
| `charts.py` | Creates four graphs to visualize the results |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# 1. (Optional) Regenerate the dataset from the OECD API.
#    A pre-fetched oecd_panel.csv is already included, so this step
#    can be skipped unless you want fresh/updated data.
python3 data.py

# 2. Train both architectures, multi-seed, with early stopping.
#    Prints per-epoch loss, per-country MAE vs. baseline, and a final
#    aggregate summary across seeds.
python3 training.py

# 3. Analyze which countries consistently beat/lose to baseline, and why.
python3 country_failure_analysis.py
```

## Method at a glance

- **Data**: OECD SDMX API — composite leading indicator, unemployment,
  CPI, long-term interest rate, industrial production, plus GDP growth
  (also included as a lagged feature). Aggregate codes (OECD, G20, EA19, etc.)
  explicitly excluded. Countries missing the leading indicator are kept
  (zero-filled + a `CLI_missing` flag) rather than dropped, recovering
  coverage from 9 → 29 countries.
- **Windowing**: 8-quarter lookback, per-country, never crossing country
  boundaries.
- **Split**: fixed calendar-date cutoffs (not random, not per-country-relative)
  — train < 2022, val 2022–2024, test ≥ 2024 — to test temporal
  generalization specifically.
- **Models**: country identity is embedded (`nn.Embedding`) and concatenated
  onto per-timestep features inside `forward()`, not treated as a raw numeric
  input. The Transformer uses causal masking, matching the LSTM's
  strictly-left-to-right information flow for a fairer comparison.
- **Training**: shared loop, MAE loss (chosen over MSE so a few volatile
  countries don't dominate the gradient), early stopping on validation loss,
  best-checkpoint weights saved directly (not re-derived from a re-run).
- **Evaluation**: naive per-country mean-growth baseline, 5-seed aggregation,
  test set touched exactly once after all decisions were finalized on
  validation.
