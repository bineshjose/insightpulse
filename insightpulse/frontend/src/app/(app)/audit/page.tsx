"use client";

import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";
import { AgentTimeline } from "@/components/charts/timeline";
import { ExportButton } from "@/components/common/export-button";
import { StatusBadge } from "@/components/common/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RECENT_RUNS, SAMPLE_RUN, loadLastRun } from "@/lib/demo-data";
import { formatNumber, formatPercent, formatUsd } from "@/lib/utils";
import type { SurveyRunResponse } from "@/lib/types";

/**
 * Audit: agent execution timeline, provenance hash with copy, run config,
 * reproduce command, and run history.
 */
export default function AuditPage() {
  const [run, setRun] = useState<SurveyRunResponse>(SAMPLE_RUN);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const stored = loadLastRun();
    if (stored) setRun(stored);
  }, []);

  const reproduceCommand = [
    "curl -X POST http://localhost:8000/api/v1/survey/run \\",
    "  -H 'Content-Type: application/json' \\",
    `  -d '{"questions": ["${run.results[0]?.question_text ?? "…"}"],`,
    `       "cohort_size": ${run.total_responses}, "seed": 42}'`,
  ].join("\n");

  const copyHash = async () => {
    await navigator.clipboard.writeText(run.provenance_hash);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Audit Trail</h1>
        <p className="text-sm text-niq-text-secondary">
          Full provenance for run <span className="font-mono">{run.run_id}</span> — replayable
          by construction.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Provenance</CardTitle>
            <CardDescription>SHA-256 fingerprint of the run configuration.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-center gap-2">
              <code className="flex-1 truncate rounded-lg bg-niq-bg px-3 py-2 font-mono text-xs">
                {run.provenance_hash}
              </code>
              <Button variant="secondary" size="sm" onClick={copyHash} aria-label="Copy hash">
                {copied ? <Check className="h-4 w-4 text-niq-green" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <ConfigRow label="Status" value={run.status} />
            <ConfigRow label="Responses" value={formatNumber(run.total_responses)} />
            <ConfigRow label="Hallucination rate" value={formatPercent(run.hallucination_rate)} />
            <ConfigRow label="Total cost" value={formatUsd(run.total_cost_usd)} />
            <div className="pt-2">
              <ExportButton filename={`audit_${run.run_id}`} data={run} format="json" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Reproduce this run</CardTitle>
            <CardDescription>
              Same request + same seed ⇒ identical responses and provenance hash.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-lg bg-niq-navy p-4 text-xs leading-relaxed text-white">
              {reproduceCommand}
            </pre>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Agent execution timeline</CardTitle>
          <CardDescription>The 8-agent LangGraph DAG in execution order.</CardDescription>
        </CardHeader>
        <CardContent>
          <AgentTimeline trace={run.agent_trace} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Run history</CardTitle>
          <CardDescription>Sample history — live runs join the top of this table.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Survey</TableHead>
                <TableHead>Respondents</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {RECENT_RUNS.map((entry) => (
                <TableRow key={entry.survey}>
                  <TableCell className="font-semibold">{entry.survey}</TableCell>
                  <TableCell>{formatNumber(entry.respondents)}</TableCell>
                  <TableCell className="font-mono text-xs">{entry.model}</TableCell>
                  <TableCell>
                    <StatusBadge status="healthy" label="Completed" />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-niq-border pb-2 last:border-0">
      <span className="text-niq-text-secondary">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
