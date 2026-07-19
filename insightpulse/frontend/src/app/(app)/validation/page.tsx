"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Finding, ResultsTable } from "@/components/experiments/shared";
import acceptanceCriteria from "@/lib/experiments/acceptance_criteria.json";
import { benchmarkValidation, sotaComparison } from "@/lib/experiments-data";

/**
 * Validation: the six acceptance criteria with pass margins, benchmark
 * cross-validation, the state-of-the-art summary, and the overall grade.
 * All values come from the recorded evaluation results (same JSON files
 * as the analyst dashboard).
 */
export default function ValidationPage() {
  const allPass = acceptanceCriteria.passed === acceptanceCriteria.total;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Validation</h1>
        <p className="text-sm text-niq-text-secondary">
          Acceptance criteria, benchmark cross-validation, and comparison with
          prior approaches.
        </p>
      </div>

      {/* Overall grade banner */}
      <div
        className={
          allPass
            ? "flex flex-wrap items-center gap-4 rounded-lg border border-niq-green bg-niq-green/10 px-5 py-4"
            : "flex flex-wrap items-center gap-4 rounded-lg border border-niq-red bg-niq-red/10 px-5 py-4"
        }
      >
        <div className={`text-3xl font-extrabold ${allPass ? "text-green-800" : "text-red-800"}`}>
          Grade {acceptanceCriteria.grade}
        </div>
        <div className={`text-sm ${allPass ? "text-green-900" : "text-red-900"}`}>
          <span className="font-bold">
            {acceptanceCriteria.passed}/{acceptanceCriteria.total} acceptance criteria passed.
          </span>{" "}
          {acceptanceCriteria.finding}
        </div>
      </div>

      {/* Acceptance criteria cards */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {acceptanceCriteria.criteria.map((criterion) => {
          const op = criterion.direction === "min" ? "≥" : "≤";
          return (
            <Card key={criterion.metric} className="border-l-4 border-l-niq-green">
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-sm">{criterion.metric}</CardTitle>
                  <CheckCircle2 aria-label="pass" className="h-5 w-5 shrink-0 text-niq-green" />
                </div>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-niq-text-secondary">Measured</span>
                  <span className="font-bold text-niq-navy">
                    {criterion.value}
                    {criterion.unit}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-niq-text-secondary">Target</span>
                  <span className="font-semibold">
                    {op} {criterion.threshold}
                    {criterion.unit}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-niq-text-secondary">Margin</span>
                  <span className="font-semibold text-green-700">
                    {criterion.margin}
                    {criterion.unit}
                  </span>
                </div>
                <p className="border-t border-niq-border pt-2 text-xs leading-relaxed text-niq-text-secondary">
                  <b>{criterion.component}.</b> {criterion.basis}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Benchmark cross-validation */}
      <Card>
        <CardHeader>
          <CardTitle>Benchmark cross-validation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <ResultsTable rows={benchmarkValidation.benchmarks} />
          <Finding>{benchmarkValidation.finding}</Finding>
        </CardContent>
      </Card>

      {/* SotA summary */}
      <Card>
        <CardHeader>
          <CardTitle>Comparison with prior approaches</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <ResultsTable
            rows={sotaComparison.results}
            isHighlighted={(row) => String(row.method).includes("InsightPulse")}
          />
          <p className="text-xs text-niq-text-secondary">{sotaComparison.note}</p>
          <Finding>{sotaComparison.finding}</Finding>
        </CardContent>
      </Card>

      {/* Failure indicator kept for completeness when a criterion regresses */}
      {!allPass && (
        <div className="flex items-center gap-2 text-sm text-red-800">
          <XCircle className="h-4 w-4" /> One or more criteria failed — see cards above.
        </div>
      )}
    </div>
  );
}
