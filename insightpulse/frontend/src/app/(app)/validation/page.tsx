"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { MetricsCard } from "@/components/charts/metrics-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { VALIDATION_CHECKS } from "@/lib/demo-data";

/**
 * Validation: thesis targets vs measured values (pass/fail cards), metric
 * rationale panels, benchmark citations, and an overall letter grade.
 */
export default function ValidationPage() {
  const passed = VALIDATION_CHECKS.filter((check) => check.pass).length;
  const ratio = passed / VALIDATION_CHECKS.length;
  const grade = ratio >= 0.9 ? "A" : ratio >= 0.75 ? "B" : "C";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-niq-navy">Validation</h1>
          <p className="text-sm text-niq-text-secondary">
            Thesis acceptance criteria vs measured results (demo ground truth: the
            empirical response bank; production: Pew ATP / ESS waves).
          </p>
        </div>
        <MetricsCard
          label="Overall validation score"
          value={`Grade ${grade}`}
          delta={`${passed}/${VALIDATION_CHECKS.length} checks passed`}
          status={grade === "A" ? "good" : grade === "B" ? "warn" : "bad"}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {VALIDATION_CHECKS.map((check) => (
          <Card key={check.metric} className={check.pass ? "border-l-4 border-l-niq-green" : "border-l-4 border-l-niq-red"}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{check.metric}</CardTitle>
                {check.pass ? (
                  <CheckCircle2 aria-label="pass" className="h-5 w-5 text-niq-green" />
                ) : (
                  <XCircle aria-label="fail" className="h-5 w-5 text-niq-red" />
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-niq-text-secondary">Thesis target</span>
                <span className="font-semibold">{check.target}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-niq-text-secondary">Measured</span>
                <span className="font-bold text-niq-navy">{check.actual}</span>
              </div>
              <p className="border-t border-niq-border pt-2 text-xs leading-relaxed text-niq-text-secondary">
                <b>Why this metric:</b> {check.why}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>External benchmark integration</CardTitle>
          <CardDescription>
            Benchmark files drop into <code>data/benchmarks/</code> with the same schema as
            the survey response bank; validation runs unchanged against them.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
          <BenchmarkCard
            name="Pew American Trends Panel"
            role="US attitudinal ground truth (wave-matched questions, 2023-2025)"
            status="mapping questions"
          />
          <BenchmarkCard
            name="European Social Survey"
            role="Cross-country attitudinal validation (ESS round 11)"
            status="mapping questions"
          />
          <BenchmarkCard
            name="Twin-2K-500"
            role="Published digital-twin benchmark for direct comparison"
            status="format compatible"
          />
        </CardContent>
      </Card>
    </div>
  );
}

function BenchmarkCard({ name, role, status }: { name: string; role: string; status: string }) {
  return (
    <div className="rounded-lg border border-niq-border bg-niq-bg p-3">
      <div className="font-bold text-niq-navy">{name}</div>
      <p className="mt-1 text-xs text-niq-text-secondary">{role}</p>
      <p className="mt-2 text-xs font-semibold text-niq-blue">{status}</p>
    </div>
  );
}
