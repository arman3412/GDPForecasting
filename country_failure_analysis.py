"""
country_failure_analysis.py

Cross-references each country's "beats baseline?" result against two
candidate explanations: GDP growth volatility (std) and training history
length (n_quarters). Run this *after* training.py -- it doesn't retrain
anything, it just reuses the same train split preprocessing.py already
builds, so there's no duplicated cutoff-date logic to drift out of sync.

NOTE: LSTM_MEAN_RATIO below is hardcoded from the 5-seed aggregate summary
already produced by training.py. If you rerun the sweep (different seeds,
hyperparameters, architecture changes), update this dict from that run's
"Aggregate Summary" table, this script does not recompute it for you.
"""

import numpy as np
import pandas as pd

import preprocessing

# Mean model_MAE/baseline_MAE ratio per country, LSTM, averaged across
# seeds [0, 1, 2, 3, 5] with early stopping -- copied from training.py's
# printed "Aggregate Summary" table.
LSTM_MEAN_RATIO = {
    "AUT": 0.77, "BEL": 1.42, "CAN": 0.71, "CZE": 0.58, "DEU": 0.84,
    "DNK": 1.02, "ESP": 0.91, "FIN": 0.80, "FRA": 1.60, "GBR": 0.69,
    "GRC": 0.76, "HUN": 0.77, "IRL": 0.82, "ISR": 0.96, "ITA": 0.98,
    "KOR": 0.90, "LTU": 0.87, "LUX": 0.85, "LVA": 0.57, "NLD": 0.60,
    "NOR": 0.94, "POL": 0.87, "PRT": 1.21, "SVK": 1.22, "SVN": 1.61,
    "SWE": 0.95, "USA": 1.70,
}


def build_country_stats():
    """Per-country n_quarters and std of GDP growth, computed from the
    train split only."""
    data_splits = preprocessing.split_data()
    df = pd.DataFrame({
        "country": data_splits["groups_train"],
        "growth": data_splits["y_train"],
    })
    stats = (
        df.groupby("country")["growth"]
        .agg(n_quarters="count", std="std")
        .reset_index()
    )
    return stats


def main():
    stats = build_country_stats()
    stats["ratio"] = stats["country"].map(LSTM_MEAN_RATIO)
    stats["result"] = np.where(stats["ratio"] < 1.0, "WIN", "LOSE")

    missing = stats[stats["ratio"].isna()]
    if len(missing):
        print("WARNING: no ratio found for these countries -- check "
              "LSTM_MEAN_RATIO against your latest training.py output:")
        print(missing["country"].tolist())
        stats = stats.dropna(subset=["ratio"])

    stats = stats.sort_values("ratio", ascending=False).reset_index(drop=True)

    # Full table, winners and losers together 
    print(f"{'Country':<10}{'Result':<8}{'Ratio':>8}{'n_quarters':>13}{'std':>10}")
    print("-" * 49)
    for _, row in stats.iterrows():
        print(f"{row['country']:<10}{row['result']:<8}{row['ratio']:>8.2f}"
              f"{row['n_quarters']:>13.0f}{row['std']:>10.3f}")

   
    corr_std = np.corrcoef(stats["std"], stats["ratio"])[0, 1]
    corr_n = np.corrcoef(stats["n_quarters"], stats["ratio"])[0, 1]

    print(f"\nCorrelation(ratio, std)        = {corr_std:+.3f}")
    print(f"Correlation(ratio, n_quarters) = {corr_n:+.3f}")

    # ---- Split LOSE group by std to check the two-failure-mode theory --
    losers = stats[stats["result"] == "LOSE"]
    winners = stats[stats["result"] == "WIN"]
    print(f"\nLOSE group   -- mean std: {losers['std'].mean():.3f}  "
          f"mean n_quarters: {losers['n_quarters'].mean():.1f}")
    print(f"WIN  group   -- mean std: {winners['std'].mean():.3f}  "
          f"mean n_quarters: {winners['n_quarters'].mean():.1f}")


if __name__ == "__main__":
    main()