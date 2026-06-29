"""
Generate the four key figures for the LSTM vs Transformer comparison.

Figure 1 — Train vs val loss curves (seed 0, both architectures)
Figure 2 — Per-country val MAE vs baseline, with 5-seed error bars
Figure 3 — Per-country test MAE vs baseline (seed 0, no error bars)
Figure 4 — Scatter: mean val ratio vs GDP_target std, per country
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from training import train

SEEDS       = [0, 1, 2, 3, 5]
REP_SEED    = 0          # representative seed for figures 1 & 3
TRAIN_CUT   = "2022-01-01"
OUT_DIR     = "."        # write PNGs next to this file

MODEL_COLORS   = {"lstm": "#2196F3", "transformer": "#FF5722"}  # used only for fig1 loss curves
WIN_COLOR      = "#2196F3"   # beats baseline
LOSE_COLOR     = "#EF5350"   # loses to baseline
BASELINE_COLOR = "#9E9E9E"  



def _ratio_order(per_country_mean):
    """Return countries sorted ascending by mean model/baseline ratio."""
    ratios = {c: m / b for c, (m, b) in per_country_mean.items() if b > 0}
    return sorted(ratios, key=lambda c: ratios[c])


def _gdp_std(train_cutoff=TRAIN_CUT):
    """Per-country GDP_target std on the training split."""
    df = pd.read_csv("oecd_panel.csv", parse_dates=["date"])
    train = df[df["date"] < train_cutoff]
    return train.groupby("country")["GDP_target"].std()


# training

all_results = {}   # all_results[model_type][seed] = {...}

for seed in SEEDS:
    print(f"  seed {seed}")
    r = train(seed=seed)
    for model_type, seed_dict in r.items():
        all_results.setdefault(model_type, {}).update(seed_dict)


gdp_std = _gdp_std()


#Figure 1: train vs val loss curves

fig1, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
fig1.suptitle("Train vs Validation Loss — seed 0", fontsize=13, fontweight="bold")

for ax, model_type in zip(axes, ["lstm", "transformer"]):
    info  = all_results[model_type][REP_SEED]
    train_h = info["train_loss_history"]
    val_h   = info["val_loss_history"]
    epochs  = range(len(train_h))
    stop_ep = info["early_stop_epoch"]
    best_ep = info["best_epoch"]
    color   = MODEL_COLORS[model_type]

    ax.plot(epochs, train_h, color=color,      lw=1.8, label="Train MAE")
    ax.plot(epochs, val_h,   color=color, lw=1.8, ls="--", alpha=0.7, label="Val MAE")
    ax.axvline(stop_ep, color="black", lw=1, ls=":", label=f"Early stop (ep {stop_ep})")
    ax.axvline(best_ep, color="green",  lw=1, ls=":", alpha=0.6, label=f"Best (ep {best_ep})")

    ax.set_title(model_type.upper())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig1_loss_curves.png", dpi=150)
print("Saved fig1_loss_curves.png")


# Figure 2: per-country val MAE vs baseline (5-seed error bars)

fig2, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig2.suptitle("Per-Country Val MAE vs Naive Mean Baseline (5 seeds)", fontsize=13, fontweight="bold")

for ax, model_type in zip(axes, ["lstm", "transformer"]):
    seed_data = all_results[model_type]
    countries = sorted(next(iter(seed_data.values()))["per_country"])

    # Mean and std of model MAE across seeds; baseline is deterministic
    mean_model = {}
    std_model  = {}
    baseline   = {}
    for c in countries:
        maes = [seed_data[s]["per_country"][c][0]
                for s in SEEDS if c in seed_data[s]["per_country"]]
        mean_model[c] = np.mean(maes)
        std_model[c]  = np.std(maes)
        baseline[c]   = seed_data[SEEDS[0]]["per_country"][c][1]

    # Sort by ratio ascending (winners left, losers right)
    ordered = sorted(countries, key=lambda c: mean_model[c] / baseline[c])
    x = np.arange(len(ordered))
    w = 0.35

    model_vals    = [mean_model[c] for c in ordered]
    model_errs    = [std_model[c]  for c in ordered]
    baseline_vals = [baseline[c]   for c in ordered]
    colors        = [WIN_COLOR if mean_model[c] < baseline[c] else LOSE_COLOR
                     for c in ordered]

    ax.bar(x - w/2, model_vals,    w, yerr=model_errs, capsize=3,
           color=colors, label="Model MAE", error_kw={"elinewidth": 1})
    ax.bar(x + w/2, baseline_vals, w, color=BASELINE_COLOR, alpha=0.55, label="Baseline MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha="right", fontsize=8)
    ax.set_title(model_type.upper())
    ax.set_ylabel("MAE")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)

    win_patch  = mpatches.Patch(color=WIN_COLOR,      label="Model MAE (beats baseline)")
    lose_patch = mpatches.Patch(color=LOSE_COLOR,     label="Model MAE (loses to baseline)")
    base_patch = mpatches.Patch(color=BASELINE_COLOR, alpha=0.55, label="Baseline MAE")
    ax.legend(handles=[win_patch, lose_patch, base_patch], fontsize=8)

fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig2_val_per_country.png", dpi=150)
print("Saved fig2_val_per_country.png")


# Figure 3: per-country test MAE vs baseline (seed 0)

fig3, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig3.suptitle(f"Per-Country Test MAE vs Naive Mean Baseline (seed {REP_SEED})", fontsize=13, fontweight="bold")

for ax, model_type in zip(axes, ["lstm", "transformer"]):
    info = all_results[model_type][REP_SEED]
    test_pc = info["test_per_country"]

    ordered = sorted(test_pc, key=lambda c: test_pc[c][0] / test_pc[c][1])
    x = np.arange(len(ordered))
    w = 0.35

    model_vals    = [test_pc[c][0] for c in ordered]
    baseline_vals = [test_pc[c][1] for c in ordered]
    colors        = [WIN_COLOR if test_pc[c][0] < test_pc[c][1] else LOSE_COLOR
                     for c in ordered]

    ax.bar(x - w/2, model_vals,    w, color=colors,         label="Model MAE")
    ax.bar(x + w/2, baseline_vals, w, color=BASELINE_COLOR, alpha=0.55, label="Baseline MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha="right", fontsize=8)
    ax.set_title(model_type.upper())
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.3)

    win_patch  = mpatches.Patch(color=WIN_COLOR,      label="Model MAE (beats baseline)")
    lose_patch = mpatches.Patch(color=LOSE_COLOR,     label="Model MAE (loses to baseline)")
    base_patch = mpatches.Patch(color=BASELINE_COLOR, alpha=0.55, label="Baseline MAE")
    ax.legend(handles=[win_patch, lose_patch, base_patch], fontsize=8)

fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig3_test_per_country.png", dpi=150)
print("Saved fig3_test_per_country.png")


# Figure 4: scatter: mean val ratio vs GDP_target std

fig4, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
fig4.suptitle("Val Ratio (model MAE / baseline) vs GDP Growth Volatility",
              fontsize=13, fontweight="bold")

for ax, model_type in zip(axes, ["lstm", "transformer"]):
    seed_data = all_results[model_type]
    countries = sorted(next(iter(seed_data.values()))["per_country"])

    ratios = {}
    for c in countries:
        maes = [seed_data[s]["per_country"][c][0]
                for s in SEEDS if c in seed_data[s]["per_country"]]
        base = seed_data[SEEDS[0]]["per_country"][c][1]
        ratios[c] = np.mean(maes) / base if base > 0 else float("nan")

    xs, ys, labels, colors = [], [], [], []
    for c in countries:
        if c not in gdp_std or np.isnan(ratios.get(c, float("nan"))):
            continue
        xs.append(gdp_std[c])
        ys.append(ratios[c])
        labels.append(c)
        colors.append(LOSE_COLOR if ratios[c] > 1 else WIN_COLOR)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="Ratio = 1 (break-even)")
    ax.scatter(xs, ys, c=colors, s=60, zorder=3)

    # Annotate every point
    for x_, y_, label in zip(xs, ys, labels):
        ax.annotate(label, (x_, y_), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, alpha=0.85)

    ax.set_title(model_type.upper())
    ax.set_xlabel("GDP_target std (training quarters)")
    ax.set_ylabel("Mean val ratio (model / baseline)")

    win_patch  = mpatches.Patch(color=WIN_COLOR,  label="Model wins (ratio < 1)")
    lose_patch = mpatches.Patch(color=LOSE_COLOR, label="Model loses (ratio > 1)")
    ax.legend(handles=[win_patch, lose_patch], fontsize=8)
    ax.grid(alpha=0.3)

fig4.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig4_ratio_vs_std.png", dpi=150)
print("Saved fig4_ratio_vs_std.png")

plt.show()
