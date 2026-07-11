"use client";

import { BarChart3 } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { DemographicBreakdown } from "@/components/charts/demographic-breakdown";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { MetricsCard } from "@/components/charts/metrics-card";
import { EmptyState } from "@/components/common/empty-state";
import { ExportButton } from "@/components/common/export-button";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { SAMPLE_RUN, loadLastRun } from "@/lib/demo-data";
import { formatPercent } from "@/lib/utils";
import type { SurveyRunResponse } from "@/lib/types";

/**
 * Results: run selector, distribution chart (raw vs calibrated), the metric
 * panel, improvement badges, demographic breakdowns, exports.
 */
export default function ResultsPage() {
  const [liveRun, setLiveRun] = useState<SurveyRunResponse | null>(null);
  const [selected, setSelected] = useState<"live" | "sample">("sample");

  useEffect(() => {
    const stored = loadLastRun();
    if (stored) {
      setLiveRun(stored);
      setSelected("live");
    }
  }, []);

  const run = selected === "live" && liveRun ? liveRun : SAMPLE_RUN;
  const result = run.results[0];

  if (!result) {
    return (
      <EmptyState
        icon={BarChart3}
        title="No results yet"
        message="Run a survey to get started — results, metrics, and breakdowns will appear here."
        action={
          <Button>
            <Link href="/survey">Run a survey</Link>
          </Button>
        }
      />
    );
  }

  const metrics = result.calibration_metrics;
  const runOptions = [
    ...(liveRun ? [{ value: "live", label: `Live run ${liveRun.run_id}` }] : []),
    { value: "sample", label: "Sample run (Organic Labeling, 250 respondents)" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-niq-navy">Survey Results</h1>
          <p className="text-sm text-niq-text-secondary">
            Raw vs calibrated distributions with the full evaluation metric suite.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            aria-label="Select run"
            className="w-72"
            options={runOptions}
            value={selected}
            onChange={(event) => setSelected(event.target.value as "live" | "sample")}
          />
          <ExportButton filename={`run_${run.run_id}_results`} data={run.results} format="json" />
          <ExportButton
            filename={`run_${run.run_id}_distribution`}
            data={result.distribution}
            format="csv"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricsCard
          label="JS divergence"
          value={metrics ? metrics.js_divergence_after.toFixed(4) : "—"}
          delta={metrics ? `was ${metrics.js_divergence_before.toFixed(4)} raw` : undefined}
          status="good"
        />
        <MetricsCard
          label="Wasserstein"
          value={metrics ? metrics.wasserstein_after.toFixed(4) : "—"}
          delta={metrics ? `was ${metrics.wasserstein_before.toFixed(4)} raw` : undefined}
          status="good"
        />
        <MetricsCard label="Shannon entropy" value={`${result.entropy.toFixed(2)} bits`} delta="diversity check" status="info" />
        <MetricsCard
          label="Hallucination rate"
          value={formatPercent(run.hallucination_rate)}
          delta="target < 5%"
          status={run.hallucination_rate < 0.05 ? "good" : "bad"}
        />
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center gap-2">
            <CardTitle>{result.question_text}</CardTitle>
            {metrics && (
              <Badge variant="green">↑ {metrics.js_improvement_pct.toFixed(0)}% improvement</Badge>
            )}
          </div>
          <CardDescription>
            {result.valid_responses} valid of {result.total_responses} responses — calibrated with
            Sinkhorn optimal transport{metrics?.iterations ? ` (${metrics.iterations} iterations)` : ""}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DistributionChart result={result} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Demographic breakdown</CardTitle>
          <CardDescription>Answer shares sliced by panel dimensions.</CardDescription>
        </CardHeader>
        <CardContent>
          <DemographicBreakdown />
        </CardContent>
      </Card>
    </div>
  );
}
