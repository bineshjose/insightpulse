"use client";

import { GitCompareArrows, X } from "lucide-react";
import { Fragment, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ConvergenceChart } from "@/components/charts/convergence-chart";
import { MetricsCard } from "@/components/charts/metrics-card";
import { ModelRadarChart } from "@/components/charts/radar-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  CHART_SERIES,
  DRIFT_SERIES,
  DRIFT_TRIGGER,
  EXPERIMENT_PARAMS,
  EXPERIMENT_RUNS,
  MODEL_COMPARISON,
  SEQUENTIAL_SUMMARY,
  type ExperimentRun,
} from "@/lib/demo-data";
import { formatPercent } from "@/lib/utils";

const EPSILONS = ["0.01", "0.05", "0.1", "0.5"] as const;

/** Experiments hub: tabbed access to the four experiment tracks + run history. */
export default function ExperimentsPage() {
  const [activeEpsilons, setActiveEpsilons] = useState<string[]>(["0.01", "0.1", "0.5"]);

  const toggleEpsilon = (eps: string) =>
    setActiveEpsilons((current) =>
      current.includes(eps) ? current.filter((e) => e !== eps) : [...current, eps],
    );

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Experiments</h1>
      </div>

      <Tabs defaultValue="llm">
        <TabsList>
          <TabsTrigger value="llm">Multi-LLM comparison</TabsTrigger>
          <TabsTrigger value="convergence">Calibration convergence</TabsTrigger>
          <TabsTrigger value="drift">Drift detection</TabsTrigger>
          <TabsTrigger value="sequential">Sequential dependency</TabsTrigger>
          <TabsTrigger value="history">Run history</TabsTrigger>
        </TabsList>

        <TabsContent value="llm">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {MODEL_COMPARISON.map((model) => (
              <Card key={model.model}>
                <CardHeader>
                  <CardTitle className="font-mono text-sm">{model.model}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1.5 text-sm">
                  <MetricRow label="JS divergence" value={model.jsDivergence.toFixed(4)} />
                  <MetricRow label="Wasserstein" value={model.wasserstein.toFixed(4)} />
                  <MetricRow label="Hallucination" value={formatPercent(model.hallucination)} />
                  <MetricRow label="Consistency" value={formatPercent(model.consistency)} />
                  <MetricRow
                    label="Safety score"
                    value={model.safetyScore === 1 ? "100%" : formatPercent(model.safetyScore)}
                  />
                  <MetricRow label="Cost / run" value={model.costUsd === 0 ? "free (local)" : `$${model.costUsd.toFixed(2)}`} />
                </CardContent>
              </Card>
            ))}
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Model profile radar</CardTitle>
            </CardHeader>
            <CardContent>
              <ModelRadarChart />
            </CardContent>
          </Card>
          <ParametersCard params={EXPERIMENT_PARAMS.llm} />
        </TabsContent>

        <TabsContent value="convergence">
          <Card>
            <CardHeader>
              <CardTitle>Sinkhorn convergence by ε</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-3 flex flex-wrap gap-4">
                {EPSILONS.map((eps) => (
                  <Label key={eps} className="mb-0 flex items-center gap-2 font-normal">
                    <Checkbox
                      checked={activeEpsilons.includes(eps)}
                      onChange={() => toggleEpsilon(eps)}
                    />
                    ε = {eps}
                  </Label>
                ))}
              </div>
              <ConvergenceChart epsilons={activeEpsilons} />
            </CardContent>
          </Card>
          <ParametersCard params={EXPERIMENT_PARAMS.convergence} />
        </TabsContent>

        <TabsContent value="drift">
          <Card>
            <CardHeader>
              <CardTitle>Category-mix drift vs 3-month baseline</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={[...DRIFT_SERIES]}>
                  <CartesianGrid vertical={false} stroke="#E5E7EB" />
                  <XAxis dataKey="month" tick={{ fill: "#6B7280", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <ReferenceLine
                    y={DRIFT_TRIGGER}
                    stroke="#6B7280"
                    strokeDasharray="5 4"
                    label={{ value: "retrain trigger", fill: "#6B7280", fontSize: 11, position: "insideTopLeft" }}
                  />
                  <Line dataKey="stationary" name="Stationary panel" stroke={CHART_SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
                  <Line dataKey="drifted" name="Injected drift" stroke={CHART_SERIES[4]} strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <ParametersCard params={EXPERIMENT_PARAMS.drift} />
        </TabsContent>

        <TabsContent value="sequential">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <MetricsCard label="Spearman — independent" value={SEQUENTIAL_SUMMARY.spearman.independent.toFixed(2)} delta="demographic floor" status="info" />
            <MetricsCard label="Spearman — conditioned" value={SEQUENTIAL_SUMMARY.spearman.conditioned.toFixed(2)} delta="prior answers in prompt" status="good" />
            <MetricsCard
              label="Contradiction rate"
              value={formatPercent(SEQUENTIAL_SUMMARY.contradictions.conditioned)}
              delta={`was ${formatPercent(SEQUENTIAL_SUMMARY.contradictions.independent)} independent`}
              status="good"
            />
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Within-person consistency by generation strategy</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={[
                    { strategy: "Independent", spearman: SEQUENTIAL_SUMMARY.spearman.independent, contradictions: SEQUENTIAL_SUMMARY.contradictions.independent * 100 },
                    { strategy: "Conditioned", spearman: SEQUENTIAL_SUMMARY.spearman.conditioned, contradictions: SEQUENTIAL_SUMMARY.contradictions.conditioned * 100 },
                    { strategy: "Empirical", spearman: SEQUENTIAL_SUMMARY.spearman.empirical, contradictions: SEQUENTIAL_SUMMARY.contradictions.empirical * 100 },
                  ]}
                  barGap={2}
                >
                  <CartesianGrid vertical={false} stroke="#E5E7EB" />
                  <XAxis dataKey="strategy" tick={{ fill: "#6B7280", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#6B7280", fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="spearman" name="Spearman ρ" fill={CHART_SERIES[0]} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="contradictions" name="Contradictions (%)" fill={CHART_SERIES[1]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
          <ParametersCard params={EXPERIMENT_PARAMS.sequential} />
        </TabsContent>

        <TabsContent value="history">
          <RunHistory />
        </TabsContent>
      </Tabs>
    </div>
  );
}

/** Run tracker: expandable rows, two-run selection, side-by-side compare. */
function RunHistory() {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparing, setComparing] = useState(false);

  const toggleSelected = (id: string) => {
    setComparing(false);
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((s) => s !== id) : [...current, id],
    );
  };

  const compareRuns = EXPERIMENT_RUNS.filter((run) => selectedIds.includes(run.id));

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle>Experiment run history</CardTitle>
          <CardDescription>
            Click a row for full result details. Select two runs to compare side by side.
          </CardDescription>
        </div>
        <Button
          size="sm"
          disabled={selectedIds.length !== 2}
          onClick={() => setComparing(true)}
        >
          <GitCompareArrows className="h-4 w-4" /> Compare ({selectedIds.length}/2)
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10" aria-label="Select" />
              <TableHead>Experiment</TableHead>
              <TableHead>Model(s)</TableHead>
              <TableHead>Cohort</TableHead>
              <TableHead>Timestamp</TableHead>
              <TableHead>Key Result</TableHead>
              <TableHead>Parameters</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {EXPERIMENT_RUNS.map((run) => {
              const expanded = expandedId === run.id;
              return (
                <Fragment key={run.id}>
                  <TableRow
                    className="cursor-pointer hover:bg-niq-bg"
                    aria-expanded={expanded}
                    onClick={() => setExpandedId(expanded ? null : run.id)}
                  >
                    <TableCell onClick={(event) => event.stopPropagation()}>
                      <Checkbox
                        aria-label={`Select ${run.experiment}`}
                        checked={selectedIds.includes(run.id)}
                        onChange={() => toggleSelected(run.id)}
                      />
                    </TableCell>
                    <TableCell className="font-semibold">{run.experiment}</TableCell>
                    <TableCell className="font-mono text-xs">{run.models}</TableCell>
                    <TableCell>{run.cohort}</TableCell>
                    <TableCell className="whitespace-nowrap text-niq-text-secondary">
                      {run.timestamp}
                    </TableCell>
                    <TableCell className="font-semibold text-niq-navy">{run.keyResult}</TableCell>
                    <TableCell className="font-mono text-xs">{run.parameters}</TableCell>
                  </TableRow>
                  {expanded && (
                    <TableRow>
                      <TableCell colSpan={7} className="bg-niq-bg">
                        <RunDetails run={run} />
                      </TableCell>
                    </TableRow>
                  )}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>

        {comparing && compareRuns.length === 2 && (
          <div className="rounded-xl border border-niq-border bg-niq-bg p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-bold text-niq-navy">Side-by-side comparison</span>
              <Button
                variant="ghost"
                size="sm"
                aria-label="Close comparison"
                onClick={() => setComparing(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {compareRuns.map((run) => (
                <div key={run.id} className="rounded-lg border border-niq-border bg-white p-4">
                  <div className="font-bold text-niq-navy">{run.experiment}</div>
                  <div className="text-xs text-niq-text-secondary">
                    {run.timestamp} · cohort {run.cohort} · {run.parameters}
                  </div>
                  <div className="mt-3">
                    <RunDetails run={run} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RunDetails({ run }: { run: ExperimentRun }) {
  return (
    <div className="space-y-1.5 py-1 text-sm">
      {run.details.map((detail) => (
        <div
          key={detail.label}
          className="flex justify-between gap-4 border-b border-niq-border pb-1.5 last:border-0"
        >
          <span className="text-niq-text-secondary">{detail.label}</span>
          <span className="text-right font-semibold">{detail.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Reproducibility panel: the exact config behind the tab's displayed results. */
function ParametersCard({ params }: { params: Record<string, string | number | boolean> }) {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>
          Exact configuration behind the displayed results — rerunning with the
          same seed reproduces them.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 text-sm sm:grid-cols-2">
          {Object.entries(params).map(([key, value]) => (
            <div key={key} className="flex justify-between gap-4 border-b border-niq-border pb-1.5">
              <span className="font-mono text-xs text-niq-text-secondary">{key}</span>
              <span className="text-right font-semibold">{String(value)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-niq-border pb-1.5 last:border-0">
      <span className="text-niq-text-secondary">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
