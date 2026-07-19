"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Callout,
  Finding,
  ResultsTable,
  StatCard,
  SubSection,
} from "@/components/experiments/shared";
import {
  AblationWaterfall,
  CostQualityScatter,
  DriftTimeline,
  GroupedBars,
  LogConvergence,
  ModelRadar,
  MultiLine,
  SelectableBars,
  TimingBars,
} from "@/components/experiments/exp-charts";
import {
  ablationStudy,
  bdclBeforeAfter,
  behaviouralWeight,
  chunkingStrategy,
  clusteringAlgorithm,
  driftDetection,
  embeddingDimension,
  encoderArchitecture,
  epsilonSensitivity,
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
} from "@/lib/experiments-data";

/** Experiments hub: the ten research evaluation tracks. */
export default function ExperimentsPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Experiments</h1>
        <p className="text-sm text-niq-text-secondary">
          Embedding, generation, and calibration tuning; multi-model, ablation, drift,
          timing, and benchmark comparisons.
        </p>
      </div>

      <Tabs defaultValue="embedding">
        <div className="overflow-x-auto">
          <TabsList className="w-max min-w-full">
            <TabsTrigger value="embedding">Embedding Tuning</TabsTrigger>
            <TabsTrigger value="generation">Generation Tuning</TabsTrigger>
            <TabsTrigger value="calibration">Calibration Analysis</TabsTrigger>
            <TabsTrigger value="multimodel">Multi-Model</TabsTrigger>
            <TabsTrigger value="ablation">Ablation Study</TabsTrigger>
            <TabsTrigger value="sequential">Sequential Dependency</TabsTrigger>
            <TabsTrigger value="drift">Drift Detection</TabsTrigger>
            <TabsTrigger value="timing">Timing &amp; Scalability</TabsTrigger>
            <TabsTrigger value="sota">SotA Comparison</TabsTrigger>
            <TabsTrigger value="hyperparams">Hyperparameter Search</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="embedding"><EmbeddingTab /></TabsContent>
        <TabsContent value="generation"><GenerationTab /></TabsContent>
        <TabsContent value="calibration"><CalibrationTab /></TabsContent>
        <TabsContent value="multimodel"><MultiModelTab /></TabsContent>
        <TabsContent value="ablation"><AblationTab /></TabsContent>
        <TabsContent value="sequential"><SequentialTab /></TabsContent>
        <TabsContent value="drift"><DriftTab /></TabsContent>
        <TabsContent value="timing"><TimingTab /></TabsContent>
        <TabsContent value="sota"><SotaTab /></TabsContent>
        <TabsContent value="hyperparams"><HyperparamsTab /></TabsContent>
      </Tabs>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 1 · Embedding tuning                                            */
/* ------------------------------------------------------------------ */

function EmbeddingTab() {
  return (
    <div className="space-y-6">
      <SubSection title="Chunking strategy">
        <ResultsTable rows={chunkingStrategy.results} />
        <GroupedBars
          data={chunkingStrategy.results}
          xKey="strategy"
          series={[
            { key: "js_downstream", label: "Downstream JS divergence" },
            { key: "silhouette", label: "Silhouette" },
          ]}
        />
        <Finding>{chunkingStrategy.finding}</Finding>
      </SubSection>

      <SubSection title="Encoder architecture">
        <ResultsTable rows={encoderArchitecture.results} />
        <GroupedBars
          data={encoderArchitecture.results}
          xKey="architecture"
          series={[
            { key: "js_downstream", label: "Downstream JS divergence" },
            { key: "silhouette", label: "Silhouette" },
          ]}
        />
        <Finding>{encoderArchitecture.finding}</Finding>
      </SubSection>

      <SubSection title="Embedding dimension">
        <ResultsTable rows={embeddingDimension.results} />
        <MultiLine
          data={embeddingDimension.results}
          xKey="dim"
          xLabel="Embedding dimension d"
          selectedX={128}
          series={[
            { key: "js_downstream", label: "Downstream JS divergence" },
            { key: "silhouette", label: "Silhouette" },
          ]}
        />
        <Finding>{embeddingDimension.finding}</Finding>
      </SubSection>

      <SubSection title="Clustering algorithm">
        <ResultsTable rows={clusteringAlgorithm.results} />
        <GroupedBars
          data={clusteringAlgorithm.results}
          xKey="algorithm"
          series={[
            { key: "silhouette", label: "Silhouette" },
            { key: "js_downstream", label: "Downstream JS divergence" },
          ]}
        />
        <Finding>{clusteringAlgorithm.finding}</Finding>
      </SubSection>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 2 · Generation tuning                                           */
/* ------------------------------------------------------------------ */

function GenerationTab() {
  return (
    <div className="space-y-6">
      <SubSection title="Prompting strategy">
        <ResultsTable rows={promptingStrategy.results} />
        <GroupedBars
          data={promptingStrategy.results}
          xKey="strategy"
          series={[
            { key: "hallucination", label: "Hallucination (%)" },
            { key: "consistency", label: "Consistency (%)" },
          ]}
        />
        <Finding>{promptingStrategy.finding}</Finding>
      </SubSection>

      <SubSection title="Temperature sweep">
        <ResultsTable rows={temperatureSweep.results} />
        <MultiLine
          data={temperatureSweep.results}
          xKey="temperature"
          xLabel="Temperature"
          selectedX={0.7}
          rightAxisKeys={["consistency"]}
          series={[
            { key: "entropy", label: "Shannon entropy (bits)" },
            { key: "hallucination", label: "Hallucination (%)" },
            { key: "consistency", label: "Consistency (%, right axis)" },
          ]}
        />
        <Finding>{temperatureSweep.finding}</Finding>
      </SubSection>

      <SubSection title="Retrieval strategy">
        <ResultsTable rows={retrievalStrategy.results} />
        <SelectableBars
          data={retrievalStrategy.results}
          xKey="strategy"
          yKey="coupling"
          yLabel="Coupling coefficient"
        />
        <Finding>{retrievalStrategy.finding}</Finding>
      </SubSection>

      <SubSection title="Response parsing">
        <ResultsTable rows={responseParsing.results} />
        <SelectableBars
          data={responseParsing.results}
          xKey="method"
          yKey="parse_rate"
          yLabel="Parse rate (%)"
        />
        <Finding>{responseParsing.finding}</Finding>
      </SubSection>

      <SubSection title="Fine-tuning vs prompt conditioning">
        <ResultsTable rows={finetuningComparison.results} />
        <GroupedBars
          data={finetuningComparison.results}
          xKey="method"
          series={[
            { key: "js", label: "JS divergence" },
            { key: "hallucination", label: "Hallucination (%)" },
          ]}
        />
        <MultiLine
          data={finetuningComparison.results}
          xKey="train_hours"
          xLabel="Training time (hours)"
          series={[{ key: "cost_usd", label: "Training cost (USD)" }]}
          height={240}
        />
        <Finding>{finetuningComparison.finding}</Finding>
      </SubSection>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 3 · Calibration analysis                                        */
/* ------------------------------------------------------------------ */

function CalibrationTab() {
  const { before, after } = bdclBeforeAfter.results;
  return (
    <div className="space-y-6">
      <SubSection title="BDCL calibration impact">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="JS divergence" value={String(after.js)} sub={`was ${before.js}`} tone="good" />
          <StatCard label="Wasserstein" value={String(after.wasserstein)} sub={`was ${before.wasserstein}`} tone="good" />
          <StatCard label="Hallucination" value={`${after.hallucination}%`} sub={`was ${before.hallucination}%`} tone="good" />
          <StatCard label="Shannon entropy" value={`${after.entropy} bits`} sub={`was ${before.entropy} bits`} tone="good" />
        </div>
        <GroupedBars
          data={[
            { phase: "Before BDCL", js: before.js, wasserstein: before.wasserstein },
            { phase: "After BDCL", js: after.js, wasserstein: after.wasserstein },
          ]}
          xKey="phase"
          series={[
            { key: "js", label: "JS divergence" },
            { key: "wasserstein", label: "Wasserstein distance" },
          ]}
          height={280}
        />
        <Finding>{bdclBeforeAfter.finding}</Finding>
      </SubSection>

      <SubSection title="Sinkhorn convergence">
        <LogConvergence curves={sinkhornConvergence.curves} selected={sinkhornConvergence.selected} />
        <Finding>{sinkhornConvergence.finding}</Finding>
      </SubSection>

      <SubSection title="Regularisation sensitivity (ε)">
        <ResultsTable rows={epsilonSensitivity.results} />
        <MultiLine
          data={epsilonSensitivity.results}
          xKey="epsilon"
          xLabel="ε (log scale)"
          logX
          selectedX={0.1}
          series={[
            { key: "js", label: "JS divergence" },
            { key: "wasserstein", label: "Wasserstein" },
            { key: "coupling_entropy", label: "Coupling entropy" },
          ]}
        />
        <Finding>{epsilonSensitivity.finding}</Finding>
      </SubSection>

      <SubSection title="Behavioural weight sensitivity (λb)">
        <ResultsTable rows={behaviouralWeight.results} />
        <MultiLine
          data={behaviouralWeight.results}
          xKey="lambda_b"
          xLabel="λb"
          selectedX={0.3}
          series={[
            { key: "js", label: "JS divergence" },
            { key: "cosine", label: "Cosine similarity" },
            { key: "coupling", label: "Coupling coefficient" },
          ]}
        />
        <Finding>{behaviouralWeight.finding}</Finding>
      </SubSection>

      <SubSection title="Fairness weight sensitivity (λf)">
        <ResultsTable rows={fairnessWeight.results} />
        <MultiLine
          data={fairnessWeight.results}
          xKey="lambda_f"
          xLabel="λf"
          selectedX={0.2}
          rightAxisKeys={["js"]}
          series={[
            { key: "max_group_deviation_pp", label: "Max group deviation (pp)" },
            { key: "js", label: "JS divergence (right axis)" },
          ]}
        />
        <Finding>{fairnessWeight.finding}</Finding>
      </SubSection>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 4 · Multi-model comparison                                      */
/* ------------------------------------------------------------------ */

function MultiModelTab() {
  const rows = multiModel.results;
  const best = {
    js_infit: Math.min(...rows.map((r) => r.js_infit)),
    hallucination: Math.min(...rows.map((r) => r.hallucination)),
    consistency: Math.max(...rows.map((r) => r.consistency)),
    entropy: Math.max(...rows.map((r) => r.entropy)),
    cost_usd: Math.min(...rows.map((r) => r.cost_usd)),
    latency_sec: Math.min(...rows.map((r) => r.latency_sec)),
  };
  const mark = (value: number, key: keyof typeof best) =>
    value === best[key] ? "bg-niq-green/10 font-bold text-green-900" : "";

  return (
    <div className="space-y-6">
      <p className="text-xs text-niq-text-secondary">
        Fixed cohort of {multiModel.cohort_size} respondents, seed {multiModel.seed},{" "}
        {multiModel.survey_instrument}. {multiModel.measurement_note}
      </p>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>JS (in-fit)</TableHead>
              <TableHead>Hallucination (%)</TableHead>
              <TableHead>Consistency (%)</TableHead>
              <TableHead>Entropy (bits)</TableHead>
              <TableHead>Cost (USD)</TableHead>
              <TableHead>Latency (s)</TableHead>
              <TableHead>Tokens / response</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.model}>
                <TableCell className="font-semibold text-niq-navy">{row.model}</TableCell>
                <TableCell>{row.provider}</TableCell>
                <TableCell className={mark(row.js_infit, "js_infit")}>{row.js_infit}</TableCell>
                <TableCell className={mark(row.hallucination, "hallucination")}>{row.hallucination}</TableCell>
                <TableCell className={mark(row.consistency, "consistency")}>{row.consistency}</TableCell>
                <TableCell className={mark(row.entropy, "entropy")}>{row.entropy}</TableCell>
                <TableCell className={mark(row.cost_usd, "cost_usd")}>{row.cost_usd.toFixed(2)}</TableCell>
                <TableCell className={mark(row.latency_sec, "latency_sec")}>{row.latency_sec}</TableCell>
                <TableCell>{row.tokens_per_response}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quality profile (normalised, best = 1.0)</CardTitle>
          </CardHeader>
          <CardContent>
            <ModelRadar
              entries={rows}
              axes={[
                ["consistency", "Fidelity"],
                ["-js_infit", "Accuracy"],
                ["-hallucination", "Hallucination control"],
                ["-cost_usd", "Cost efficiency"],
                ["-latency_sec", "Throughput"],
              ]}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Cost vs quality frontier</CardTitle>
            <CardDescription>Cost per complete run against logical consistency.</CardDescription>
          </CardHeader>
          <CardContent>
            <CostQualityScatter entries={rows} />
          </CardContent>
        </Card>
      </div>
      <Finding>{multiModel.finding}</Finding>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 5 · Ablation study                                              */
/* ------------------------------------------------------------------ */

function AblationTab() {
  return (
    <div className="space-y-6">
      <Callout>
        <strong>BDCL contributes 78% of the quality improvement</strong> — the largest
        single component effect (JS 0.078 → 0.017).
      </Callout>
      <ResultsTable rows={ablationStudy.results} isHighlighted={(row) => row.config === "Full pipeline"} />
      <AblationWaterfall rows={ablationStudy.results} />
      <Finding>{ablationStudy.finding}</Finding>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 6 · Sequential dependency                                       */
/* ------------------------------------------------------------------ */

function SequentialTab() {
  const rows = [
    { strategy: "Independent", ...sequentialDependency.results.independent },
    { strategy: "Sequential conditioning", ...sequentialDependency.results.conditioned },
    { strategy: "Human panel reference", ...sequentialDependency.results.empirical },
  ];
  return (
    <div className="space-y-6">
      <Callout>
        <strong>+{sequentialDependency.improvement_pp} pp consistency improvement</strong>{" "}
        from sequential inter-question conditioning.
      </Callout>
      <ResultsTable rows={rows} isHighlighted={(row) => row.strategy === "Sequential conditioning"} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Spearman ρ on dependent question pairs</CardTitle>
          </CardHeader>
          <CardContent>
            <GroupedBars data={rows} xKey="strategy" series={[{ key: "spearman", label: "Spearman ρ" }]} height={280} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Logical contradiction rate</CardTitle>
          </CardHeader>
          <CardContent>
            <GroupedBars
              data={rows}
              xKey="strategy"
              series={[{ key: "contradiction_rate", label: "Contradictions (%)" }]}
              height={280}
            />
          </CardContent>
        </Card>
      </div>
      <Finding>{sequentialDependency.finding}</Finding>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 7 · Drift detection                                             */
/* ------------------------------------------------------------------ */

function DriftTab() {
  return (
    <div className="space-y-6">
      {driftDetection.triggered ? (
        <Callout tone="red">Retraining threshold crossed — see the retraining pipeline below.</Callout>
      ) : (
        <Callout>
          <strong>No retraining triggered</strong> across the nine-month evaluation period.
        </Callout>
      )}
      <p className="text-xs text-niq-text-secondary">Alert threshold: {driftDetection.formula}</p>
      <DriftTimeline points={driftDetection.monthly_drift} threshold={driftDetection.threshold} />
      <Finding>{driftDetection.finding}</Finding>
      <p className="text-sm">
        <span className="font-bold">Retraining pipeline (drift-triggered):</span>{" "}
        {driftDetection.retraining_pipeline.join(" → ")}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 8 · Timing & scalability                                        */
/* ------------------------------------------------------------------ */

function TimingTab() {
  const rows = timingBenchmarks.results;
  return (
    <div className="space-y-6">
      <p className="text-xs text-niq-text-secondary">Hardware: {timingBenchmarks.hardware}</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <SubSection title="Training">
          <ResultsTable
            rows={rows.filter((row) => row.category === "training")}
            hide={["category", "time_seconds"]}
          />
        </SubSection>
        <SubSection title="Inference">
          <ResultsTable
            rows={rows.filter((row) => row.category === "inference")}
            hide={["category", "time_seconds"]}
          />
        </SubSection>
      </div>
      <SubSection title="Component timings">
        <TimingBars rows={rows} />
      </SubSection>
      <SubSection title="Turnaround comparison">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Traditional wave cost" value="$4,000-8,000" sub="per wave" tone="warn" />
          <StatCard label="InsightPulse run cost" value="$2.14" sub="Claude Sonnet, 200 resp." tone="good" />
          <StatCard label="Traditional turnaround" value="~3 weeks" sub="design → field → analysis" tone="warn" />
          <StatCard label="InsightPulse turnaround" value="3.2 min" sub="1,248 respondents/min" tone="good" />
        </div>
      </SubSection>
      <Finding>{timingBenchmarks.finding}</Finding>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 9 · State-of-the-art comparison                                 */
/* ------------------------------------------------------------------ */

function SotaTab() {
  return (
    <div className="space-y-6">
      <p className="text-xs text-niq-text-secondary">{sotaComparison.note}</p>
      <ResultsTable
        rows={sotaComparison.results}
        isHighlighted={(row) => String(row.method).includes("InsightPulse")}
      />
      <GroupedBars
        data={sotaComparison.results}
        xKey="method"
        series={[
          { key: "hallucination", label: "Hallucination (%)" },
          { key: "consistency", label: "Consistency (%)" },
        ]}
        height={340}
      />
      <Finding>{sotaComparison.finding}</Finding>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 10 · Hyperparameter search                                      */
/* ------------------------------------------------------------------ */

function HyperparamsTab() {
  const parameters = Object.entries(hyperparameterTuning.parameters).map(
    ([parameter, spec]) => ({
      parameter,
      grid: spec.grid.join(", "),
      selected: String(spec.selected),
      criterion: spec.criterion,
    }),
  );
  const details: { title: string; rows: Record<string, unknown>[] }[] = [
    { title: "Sinkhorn regularisation (ε)", rows: epsilonSensitivity.results },
    { title: "Behavioural weight (λb)", rows: behaviouralWeight.results },
    { title: "Fairness weight (λf)", rows: fairnessWeight.results },
    { title: "Temperature", rows: temperatureSweep.results },
    { title: "Embedding dimension", rows: embeddingDimension.results },
    { title: "Encoder architecture", rows: encoderArchitecture.results },
    { title: "Cluster count (K)", rows: hyperparameterTuning.cluster_sweep },
    { title: "Sinkhorn iteration budget", rows: hyperparameterTuning.sinkhorn_iteration_sweep },
  ];
  return (
    <div className="space-y-6">
      <SubSection title="Parameter selection summary">
        <ResultsTable rows={parameters} isHighlighted={() => false} />
      </SubSection>
      <SubSection title="Cluster count selection (K)">
        <MultiLine
          data={hyperparameterTuning.cluster_sweep}
          xKey="K"
          xLabel="K"
          selectedX={5}
          series={[
            { key: "silhouette", label: "Silhouette" },
            { key: "js_downstream", label: "Downstream JS divergence" },
          ]}
        />
        <Finding>{hyperparameterTuning.finding}</Finding>
      </SubSection>
      <SubSection title="Full grid searches">
        <div className="space-y-3">
          {details.map((detail) => (
            <details key={detail.title} className="rounded-md border border-niq-border bg-niq-card">
              <summary className="cursor-pointer px-4 py-2.5 text-sm font-semibold text-niq-navy">
                {detail.title}
              </summary>
              <div className="border-t border-niq-border p-3">
                <ResultsTable rows={detail.rows} />
              </div>
            </details>
          ))}
        </div>
      </SubSection>
    </div>
  );
}
