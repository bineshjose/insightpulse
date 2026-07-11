"""Experiment: multi-LLM comparison (evaluator feedback #1, #6, #8).

Runs the identical survey (same questions, same cohort, same seed) across
Claude Sonnet, GPT-4o, and a local Ollama model, isolating model choice as
the only variable. Reports the full metric suite per model, before and
after BDCL calibration.

The before/after split answers evaluator feedback #6 (what changes when
switching LLMs): each model's raw distortion differs — mode collapse is
strongest on the local model — so the calibration layer does a different
amount of transport work per model. Calibration parameters are therefore
re-fit per model, never shared.

By default responses come from the offline demo engine (reproducible,
no API keys). Pass ``--live`` to route real LLM calls through the LangGraph
pipeline instead (requires API keys / a running Ollama server).

Usage:
    python -m experiments.multi_llm_comparison [--live] [--cohort-size N] [--seed N]

Outputs:
    experiments/results/multi_llm_comparison.json
    experiments/results/multi_llm_comparison.png
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import PALETTE, save_figure, save_results, setup_experiment
from insightpulse import demo_engine

# The three models under comparison (evaluator feedback #1).
MODELS = ["claude-sonnet-4-6", "gpt-4o", "ollama/llama3.1"]
MODEL_COLORS = dict(zip(MODELS, PALETTE, strict=False))

DEFAULT_COHORT_SIZE = 300
DEFAULT_SEED = 42


def run_demo(cohort_size: int, seed: int) -> list[dict[str, Any]]:
    """Run the survey across all models via the offline demo engine.

    Args:
        cohort_size: Number of panelist households per run.
        seed: Shared random seed (per-model offsets applied internally).

    Returns:
        One summary dict per model.
    """
    catalog = demo_engine.question_catalog()
    panelists = demo_engine.load_panelists()
    cohort = panelists.sample(n=cohort_size, random_state=seed)

    summaries = []
    for model in MODELS:
        run = demo_engine.run_survey(catalog, cohort, model, seed=seed)
        raw = [r["metrics_raw"] for r in run["question_results"]]
        cal = [r["metrics_calibrated"] for r in run["question_results"]]
        summaries.append({
            "model": model,
            "mode": "demo",
            "js_divergence_raw": float(np.mean([x["js_divergence"] for x in raw])),
            "js_divergence_calibrated": float(np.mean([x["js_divergence"] for x in cal])),
            "wasserstein_raw": float(np.mean([x["wasserstein_distance"] for x in raw])),
            "wasserstein_calibrated": float(
                np.mean([x["wasserstein_distance"] for x in cal])
            ),
            "shannon_entropy": float(np.mean([x["shannon_entropy"] for x in cal])),
            "hallucination_rate": run["totals"]["hallucination_rate"],
            "consistency_score": run["totals"]["consistency_score"],
            "total_cost_usd": run["totals"]["total_cost_usd"],
            "throughput_per_min": run["totals"]["throughput_per_min"],
        })
    return summaries


async def run_live(cohort_size: int, seed: int) -> list[dict[str, Any]]:
    """Run the survey across all models through the real LangGraph pipeline.

    Requires ANTHROPIC_API_KEY / OPENAI_API_KEY / a reachable Ollama server
    for the respective models; models whose provider is unreachable are
    reported with an error and skipped.

    Args:
        cohort_size: Number of panelist households per run.
        seed: Random seed passed to the pipeline.

    Returns:
        One summary dict per model (with "error" set for failed models).
    """
    from insightpulse.agents.orchestrator import run_survey

    catalog = demo_engine.question_catalog()
    questions = [q["text"] for q in catalog]

    summaries = []
    for model in MODELS:
        try:
            result = await run_survey(
                questions=questions,
                cohort_size=cohort_size,
                context="US consumer goods market",
                models=[model],
                seed=seed,
            )
            cal_metrics = result.get("calibration_metrics", [])
            summaries.append({
                "model": model,
                "mode": "live",
                "js_divergence_raw": float(np.mean(
                    [m["js_divergence_before"] for m in cal_metrics]
                )) if cal_metrics else None,
                "js_divergence_calibrated": float(np.mean(
                    [m["js_divergence_after"] for m in cal_metrics]
                )) if cal_metrics else None,
                "wasserstein_raw": float(np.mean(
                    [m["wasserstein_before"] for m in cal_metrics]
                )) if cal_metrics else None,
                "wasserstein_calibrated": float(np.mean(
                    [m["wasserstein_after"] for m in cal_metrics]
                )) if cal_metrics else None,
                "shannon_entropy": float(np.mean(
                    list(result.get("response_entropy", {}).values()) or [0.0]
                )),
                "hallucination_rate": result.get("hallucination_rate", 0.0),
                "consistency_score": result.get("consistency_score", 0.0),
                "total_cost_usd": result.get("total_cost_usd", 0.0),
                "throughput_per_min": None,
            })
        except Exception as exc:  # provider unreachable, bad key, etc.
            summaries.append({"model": model, "mode": "live", "error": str(exc)})
    return summaries


def plot_comparison(summaries: list[dict[str, Any]]) -> plt.Figure:
    """Render the 2x2 comparison figure (one metric per panel — no dual axes).

    Args:
        summaries: Per-model summary dicts (error-free entries only).

    Returns:
        The assembled matplotlib figure.
    """
    panels = [
        ("js_divergence_calibrated", "JS divergence after calibration", "{:.4f}"),
        ("wasserstein_calibrated", "Wasserstein distance after calibration", "{:.4f}"),
        ("hallucination_rate", "Hallucination rate", "{:.1%}"),
        ("consistency_score", "Logical consistency", "{:.1%}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "Multi-LLM comparison — identical survey, cohort, and seed "
        f"(n = {len(summaries)} models)",
        fontsize=12,
    )

    models = [s["model"] for s in summaries]
    for ax, (key, title, fmt) in zip(axes.flat, panels, strict=True):
        values = [s[key] for s in summaries]
        # Sub-0.005 metrics get milli-units so y-ticks stay distinct and readable.
        scale = 1000.0 if max(values) < 0.005 else 1.0
        if scale != 1.0:
            title += " (×10⁻³)"
        bars = ax.bar(
            models, [v * scale for v in values],
            color=[MODEL_COLORS[mdl] for mdl in models],
            width=0.55, zorder=2,
        )
        ax.bar_label(bars, labels=[fmt.format(v) for v in values],
                     fontsize=8.5, padding=2)
        ax.set_title(title)
        ax.margins(y=0.15)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="x", visible=False)

    lower = "lower is better"
    higher = "higher is better"
    for ax, note in zip(axes.flat, [lower, lower, lower, higher], strict=True):
        ax.set_xlabel(note, fontsize=8.5, style="italic")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> dict[str, Any]:
    """Run the multi-LLM comparison experiment end to end.

    Returns:
        The results payload written to JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Use real LLM calls via the LangGraph pipeline")
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logger = setup_experiment(__name__)
    from experiments.common import require_sample_data
    if not require_sample_data():
        raise SystemExit(1)

    logger.info(
        "multi_llm_comparison_start",
        models=MODELS, mode="live" if args.live else "demo",
        cohort_size=args.cohort_size, seed=args.seed,
    )

    if args.live:
        summaries = asyncio.run(run_live(args.cohort_size, args.seed))
    else:
        summaries = run_demo(args.cohort_size, args.seed)

    usable = [s for s in summaries if "error" not in s]
    for failed in (s for s in summaries if "error" in s):
        logger.warning("model_run_failed", model=failed["model"], error=failed["error"])

    payload = {
        "experiment": "multi_llm_comparison",
        "mode": "live" if args.live else "simulation",
        "config": {"models": MODELS, "cohort_size": args.cohort_size, "seed": args.seed},
        "results": summaries,
        "calibration_note": (
            "js_divergence_raw differs strongly across models (mode collapse is "
            "model-specific), so BDCL calibration parameters are re-fit per model "
            "— the transport work required is not transferable between LLMs "
            "(evaluator feedback #6)."
        ),
    }
    results_path = save_results("multi_llm_comparison", payload)

    figure_path = None
    if usable:
        figure_path = save_figure(plot_comparison(usable), "multi_llm_comparison")

    logger.info(
        "multi_llm_comparison_complete",
        results=str(results_path),
        figure=str(figure_path),
        models_compared=len(usable),
    )
    return payload


if __name__ == "__main__":
    main()
