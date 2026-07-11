"""Experiment: temporal drift detection and retraining trigger definition.

Addresses evaluator feedback #2 (retraining pipeline: time window, drift
detection, trigger criteria) with an empirically calibrated design:

1. **Noise floor.** On the stationary panel data, compute monthly JS
   divergence of the category-mix distribution against a 3-month baseline.
   Since the generator is stationary, this series is pure sampling noise —
   its mean + 3σ defines the retraining trigger, so the trigger is derived
   from the panel's own variability rather than hand-picked.

2. **Detection.** Inject a controlled, progressive category-mix shift into
   the final months (consumers migrating toward health-oriented categories)
   and verify the trigger fires: two consecutive months above the trigger
   level, or any single month above 2x the trigger.

Retraining action when triggered: re-train the L2 behavioral encoder,
re-cluster archetypes (K = 5), and re-fit the BDCL calibration layer.
Fallback cadence: quarterly re-embedding even without a trigger.

Usage:
    python -m experiments.drift_detection [--seed N]

Outputs:
    experiments/results/drift_detection.json
    experiments/results/drift_detection.png
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (
    PALETTE,
    require_sample_data,
    save_figure,
    save_results,
    setup_experiment,
)
from insightpulse import demo_engine
from insightpulse.utils import metrics as m

BASELINE_MONTHS = 3
# Trigger = noise-floor mean + K_SIGMA * noise-floor std (empirical, not hand-picked).
K_SIGMA = 3.0
# Trigger rule: 2 consecutive months above trigger, or 1 month above 2x trigger.
CONSECUTIVE_MONTHS = 2
HARD_MULTIPLIER = 2.0

# Injected drift: progressive migration toward health-oriented categories.
DRIFT_MONTHS = 4
DRIFT_TARGET_CATEGORIES = ("produce", "personal_care")
DRIFT_SOURCE_CATEGORIES = ("snacks", "frozen_foods")
# Fraction of source-category purchases re-labeled per drift month (cumulative).
DRIFT_RATE_PER_MONTH = 0.12

CRITICAL = "#d03b3b"
DEFAULT_SEED = 42


def monthly_category_mix(purchases: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    """Compute the category-mix count vector for every month.

    Args:
        purchases: Purchase records with a transaction_date column.

    Returns:
        Tuple of (sorted category list, month -> count vector).
    """
    frame = purchases.assign(
        month=pd.to_datetime(purchases["transaction_date"]).dt.to_period("M").astype(str)
    )
    categories = sorted(frame["product_category"].unique())
    mixes: dict[str, np.ndarray] = {}
    for month, group in frame.groupby("month"):
        counts = group["product_category"].value_counts()
        # Laplace smoothing keeps JS finite for empty categories.
        mixes[str(month)] = np.array(
            [counts.get(c, 0) for c in categories], dtype=float
        ) + 0.5
    return categories, mixes


def drift_series(mixes: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, list[float]]:
    """JS divergence of each post-baseline month against the baseline mix.

    Args:
        mixes: Month -> category count vector.

    Returns:
        Tuple of (post-baseline months, baseline vector, JS per month).
    """
    months = sorted(mixes)
    baseline = np.sum([mixes[mth] for mth in months[:BASELINE_MONTHS]], axis=0)
    post = months[BASELINE_MONTHS:]
    return post, baseline, [m.js_divergence(mixes[mth], baseline) for mth in post]


def inject_drift(purchases: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Inject a progressive category-mix shift into the final months.

    Re-labels a growing fraction of snack/frozen purchases as produce/
    personal-care purchases — the classic 'health shift' pattern the
    retraining pipeline must catch.

    Args:
        purchases: Original purchase records.
        seed: Random seed for the re-labeling choices.

    Returns:
        A drifted copy of the purchases DataFrame.
    """
    rng = np.random.default_rng(seed)
    drifted = purchases.copy()
    month = pd.to_datetime(drifted["transaction_date"]).dt.to_period("M").astype(str)
    drift_window = sorted(month.unique())[-DRIFT_MONTHS:]

    for i, target_month in enumerate(drift_window, start=1):
        fraction = min(1.0, DRIFT_RATE_PER_MONTH * i)
        candidates = drifted.index[
            (month == target_month)
            & (drifted["product_category"].isin(DRIFT_SOURCE_CATEGORIES))
        ]
        n_shift = int(len(candidates) * fraction)
        if n_shift == 0:
            continue
        chosen = rng.choice(candidates, size=n_shift, replace=False)
        drifted.loc[chosen, "product_category"] = rng.choice(
            DRIFT_TARGET_CATEGORIES, size=n_shift
        )
    return drifted


def apply_trigger_rule(
    months: list[str], values: list[float], trigger: float
) -> dict[str, Any]:
    """Evaluate the retraining trigger rule over a drift series.

    Args:
        months: Post-baseline months in order.
        values: JS divergence per month.
        trigger: The trigger level.

    Returns:
        Dict with fired flag, firing month, rule that fired, and breaches.
    """
    breaches = [mth for mth, v in zip(months, values, strict=True) if v > trigger]
    consecutive = 0
    for mth, value in zip(months, values, strict=True):
        if value > HARD_MULTIPLIER * trigger:
            return {
                "fired": True, "month": mth, "rule": "hard",
                "detail": f"JS {value:.4f} > {HARD_MULTIPLIER}x trigger", "breaches": breaches,
            }
        consecutive = consecutive + 1 if value > trigger else 0
        if consecutive >= CONSECUTIVE_MONTHS:
            return {
                "fired": True, "month": mth, "rule": "consecutive",
                "detail": f"{CONSECUTIVE_MONTHS} consecutive months above trigger",
                "breaches": breaches,
            }
    return {"fired": False, "month": None, "rule": None, "breaches": breaches}


def plot_drift(
    months: list[str],
    stationary: list[float],
    drifted: list[float],
    trigger: float,
    detection: dict[str, Any],
) -> plt.Figure:
    """Render the two-panel drift figure (stationary noise floor vs injected drift).

    Args:
        months: Post-baseline months.
        stationary: JS series on the original data.
        drifted: JS series on the drift-injected data.
        trigger: Empirical trigger level.
        detection: Trigger evaluation result for the drifted series.

    Returns:
        The assembled matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    fig.suptitle(
        "Behavioral drift monitoring — monthly category mix vs. "
        f"{BASELINE_MONTHS}-month baseline (JS divergence)",
        fontsize=12,
    )
    tick_labels = [mth[2:] for mth in months]  # YY-MM

    for ax, series, title in (
        (ax1, stationary, "Stationary panel (noise floor)"),
        (ax2, drifted, f"Injected drift in final {DRIFT_MONTHS} months"),
    ):
        ax.plot(tick_labels, series, color=PALETTE[0], linewidth=1.8,
                marker="o", markersize=4.5, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
        ax.axhline(trigger, color="#898781", linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"trigger = {trigger:.4f} (noise mean + {K_SIGMA:g}σ)",
                    xy=(0.02, trigger), xycoords=("axes fraction", "data"),
                    fontsize=8, color="#898781", va="bottom")
        ax.set_title(title)
        ax.set_xlabel("month")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.set_ylabel("JS divergence vs baseline")

    if detection["fired"]:
        idx = months.index(detection["month"])
        ax2.plot(tick_labels[idx], drifted[idx], marker="o", markersize=10,
                 color=CRITICAL, markeredgecolor="white", markeredgewidth=1.5,
                 linestyle="none", zorder=4)
        ax2.annotate(
            f"⚠ retrain triggered ({detection['rule']} rule)",
            xy=(idx, drifted[idx]), xytext=(idx - 0.4, drifted[idx] * 1.15),
            fontsize=9, color=CRITICAL, ha="right",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> dict[str, Any]:
    """Run the drift detection experiment end to end.

    Returns:
        The results payload written to JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logger = setup_experiment(__name__)
    if not require_sample_data():
        raise SystemExit(1)

    purchases = demo_engine.load_purchases()

    # 1. Noise floor on the stationary panel -> empirical trigger.
    _, mixes = monthly_category_mix(purchases)
    months, _, stationary = drift_series(mixes)
    noise_mean = float(np.mean(stationary))
    noise_std = float(np.std(stationary))
    trigger = noise_mean + K_SIGMA * noise_std

    stationary_detection = apply_trigger_rule(months, stationary, trigger)

    # 2. Injected drift -> the trigger must fire.
    _, drifted_mixes = monthly_category_mix(inject_drift(purchases, args.seed))
    drifted_months, _, drifted = drift_series(drifted_mixes)
    detection = apply_trigger_rule(drifted_months, drifted, trigger)

    logger.info(
        "drift_detection_results",
        trigger=round(trigger, 5),
        noise_mean=round(noise_mean, 5),
        noise_std=round(noise_std, 5),
        false_alarm_on_stationary=stationary_detection["fired"],
        drift_detected=detection["fired"],
        detection_month=detection["month"],
        detection_rule=detection["rule"],
    )

    payload = {
        "experiment": "drift_detection",
        "config": {
            "baseline_months": BASELINE_MONTHS,
            "k_sigma": K_SIGMA,
            "consecutive_months": CONSECUTIVE_MONTHS,
            "hard_multiplier": HARD_MULTIPLIER,
            "drift_months": DRIFT_MONTHS,
            "drift_rate_per_month": DRIFT_RATE_PER_MONTH,
            "seed": args.seed,
        },
        "noise_floor": {"mean": noise_mean, "std": noise_std, "trigger": trigger},
        "stationary_series": dict(zip(months, stationary, strict=True)),
        "drifted_series": dict(zip(drifted_months, drifted, strict=True)),
        "stationary_detection": stationary_detection,
        "drift_detection": detection,
        "retraining_pipeline": {
            "monitoring_window": "rolling 1 month vs 3-month baseline",
            "metric": "JS divergence over the category-mix distribution",
            "trigger": (
                f"JS > noise mean + {K_SIGMA:g} sigma for {CONSECUTIVE_MONTHS} consecutive "
                f"months, or any month > {HARD_MULTIPLIER:g}x that level"
            ),
            "action": (
                "re-train L2 behavioral encoder, re-cluster archetypes (K=5), "
                "re-fit BDCL calibration"
            ),
            "fallback_cadence": "quarterly re-embedding even without a trigger",
        },
    }
    results_path = save_results("drift_detection", payload)
    figure_path = save_figure(
        plot_drift(months, stationary, drifted, trigger, detection),
        "drift_detection",
    )

    logger.info(
        "drift_detection_complete",
        results=str(results_path), figure=str(figure_path),
    )
    return payload


if __name__ == "__main__":
    main()
