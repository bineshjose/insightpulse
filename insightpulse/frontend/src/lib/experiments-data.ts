/**
 * Experiment result data — imported verbatim from the JSON files copied
 * out of `src/insightpulse/analytics/experiments/results/` (the single
 * source of truth shared with the Python framework and the test suite).
 *
 * To refresh after editing the Python-side results:
 *   cp src/insightpulse/analytics/experiments/results/*.json frontend/src/lib/experiments/
 * (wired as `make sync-experiments`).
 */

import ablationStudy from "./experiments/ablation_study.json";
import bdclBeforeAfter from "./experiments/bdcl_before_after.json";
import behaviouralWeight from "./experiments/behavioural_weight.json";
import benchmarkValidation from "./experiments/benchmark_validation.json";
import chunkingStrategy from "./experiments/chunking_strategy.json";
import clusteringAlgorithm from "./experiments/clustering_algorithm.json";
import driftDetection from "./experiments/drift_detection.json";
import embeddingDimension from "./experiments/embedding_dimension.json";
import encoderArchitecture from "./experiments/encoder_architecture.json";
import epsilonSensitivity from "./experiments/epsilon_sensitivity.json";
import failureAnalysis from "./experiments/failure_analysis.json";
import fairnessWeight from "./experiments/fairness_weight.json";
import finetuningComparison from "./experiments/finetuning_comparison.json";
import hyperparameterTuning from "./experiments/hyperparameter_tuning.json";
import multiModel from "./experiments/multi_model.json";
import promptingStrategy from "./experiments/prompting_strategy.json";
import responseParsing from "./experiments/response_parsing.json";
import retrievalStrategy from "./experiments/retrieval_strategy.json";
import sequentialDependency from "./experiments/sequential_dependency.json";
import sinkhornConvergence from "./experiments/sinkhorn_convergence.json";
import sotaComparison from "./experiments/sota_comparison.json";
import temperatureSweep from "./experiments/temperature_sweep.json";
import timingBenchmarks from "./experiments/timing_benchmarks.json";

export {
  ablationStudy,
  bdclBeforeAfter,
  behaviouralWeight,
  benchmarkValidation,
  chunkingStrategy,
  clusteringAlgorithm,
  driftDetection,
  embeddingDimension,
  encoderArchitecture,
  epsilonSensitivity,
  failureAnalysis,
  fairnessWeight,
  finetuningComparison,
  hyperparameterTuning,
  multiModel,
  promptingStrategy,
  responseParsing,
  retrievalStrategy,
  sequentialDependency,
  sinkhornConvergence,
  sotaComparison,
  temperatureSweep,
  timingBenchmarks,
};

/** NIQ chart palette shared by all experiment charts. */
export const EXP_COLORS = {
  navy: "#003865",
  blue: "#00A4E4",
  green: "#6CC24A",
  amber: "#F2A900",
  red: "#E03C31",
  grey: "#9AA7B4",
  grid: "#E3E8EE",
} as const;

/** Series ramp for multi-series charts. */
export const EXP_SERIES = [
  EXP_COLORS.navy,
  EXP_COLORS.blue,
  EXP_COLORS.amber,
  EXP_COLORS.green,
  EXP_COLORS.red,
  "#7D5BA6",
  "#00877C",
  EXP_COLORS.grey,
] as const;
