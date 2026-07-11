"use client";

import { useState } from "react";
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  CHART_SERIES,
  DRIFT_SERIES,
  DRIFT_TRIGGER,
  MODEL_COMPARISON,
  SEQUENTIAL_SUMMARY,
} from "@/lib/demo-data";
import { formatPercent } from "@/lib/utils";

const EPSILONS = ["0.01", "0.05", "0.1", "0.5"] as const;

/** Experiments hub: tabbed access to the four thesis experiments. */
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
        <p className="text-sm text-niq-text-secondary">
          The four reproducible thesis experiments — values from{" "}
          <code>experiments/results/</code> (regenerate with <code>make experiments</code>).
        </p>
      </div>

      <Tabs defaultValue="llm">
        <TabsList>
          <TabsTrigger value="llm">Multi-LLM comparison</TabsTrigger>
          <TabsTrigger value="convergence">Calibration convergence</TabsTrigger>
          <TabsTrigger value="drift">Drift detection</TabsTrigger>
          <TabsTrigger value="sequential">Sequential dependency</TabsTrigger>
        </TabsList>

        <TabsContent value="llm">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
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
                  <MetricRow label="Cost / run" value={model.costUsd === 0 ? "free (local)" : `$${model.costUsd.toFixed(2)}`} />
                </CardContent>
              </Card>
            ))}
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Model profile radar</CardTitle>
              <CardDescription>All axes normalized: 100 = best observed value.</CardDescription>
            </CardHeader>
            <CardContent>
              <ModelRadarChart />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="convergence">
          <Card>
            <CardHeader>
              <CardTitle>Sinkhorn convergence by ε</CardTitle>
              <CardDescription>
                Marginal error per iteration. ε trades convergence speed against
                transport-plan sharpness — production uses ε = 0.1.
              </CardDescription>
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
        </TabsContent>

        <TabsContent value="drift">
          <Card>
            <CardHeader>
              <CardTitle>Category-mix drift vs 3-month baseline</CardTitle>
              <CardDescription>
                Trigger = noise mean + 3σ ({DRIFT_TRIGGER}). Two consecutive months above
                the line fire the retraining pipeline (re-embed → re-cluster → re-fit BDCL).
              </CardDescription>
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
              <CardDescription>
                Conditioning raises consistency well above the demographic floor while
                leaving marginal distributions unchanged.
              </CardDescription>
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
        </TabsContent>
      </Tabs>
    </div>
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
