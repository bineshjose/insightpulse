"use client";

import Link from "next/link";
import { Activity, BarChart3, CreditCard, FlaskConical, Target, TrendingUp } from "lucide-react";
import { MetricsCard } from "@/components/charts/metrics-card";
import { StatusBadge } from "@/components/common/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RECENT_RUNS } from "@/lib/demo-data";
import { useAuth } from "@/lib/auth";
import { formatNumber, formatUsd } from "@/lib/utils";

const LAYERS = [
  { code: "L1", name: "Data Layer", detail: "Panelists · Purchases · History" },
  { code: "L2", name: "Embedding", detail: "Transformer · K-Means · FAISS" },
  { code: "L3", name: "Digital Twins", detail: "Persona-prompted LLMs" },
  { code: "L4", name: "BDCL Calibration", detail: "Sinkhorn optimal transport" },
  { code: "L5", name: "Insights", detail: "Analytics · Drift · Reports" },
] as const;

/** Overview: KPI row, architecture diagram, quick actions, recent runs. */
export default function DashboardPage() {
  const { user } = useAuth();
  const creditsUsed = user ? user.creditsTotal - user.creditsBalance : 0;
  const creditsPct = user ? Math.round((creditsUsed / user.creditsTotal) * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">
          Welcome back, {user?.name.split(" ")[0]}
        </h1>
        <p className="text-sm text-niq-text-secondary">
          {new Date().toLocaleDateString("en-US", {
            weekday: "long",
            day: "numeric",
            month: "long",
            year: "numeric",
          })}{" "}
          · {user?.title}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricsCard label="Total surveys run" value="128" delta="▲ 12 this month" status="info" icon={TrendingUp} />
        <MetricsCard label="Avg calibration accuracy" value="98.3%" delta="target ≥ 90%" status="good" icon={Activity} />
        <MetricsCard label="Hallucination rate" value="1.9%" delta="target < 5%" status="good" icon={BarChart3} />
        <MetricsCard
          label="Monthly credit usage"
          value={`${formatNumber(creditsUsed)} / ${formatNumber(user?.creditsTotal ?? 0)}`}
          delta={`${creditsPct}% consumed`}
          status={creditsPct > 80 ? "warn" : "info"}
          icon={CreditCard}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System architecture</CardTitle>
          <CardDescription>
            Five layers orchestrated by an 8-agent LangGraph DAG with validation-retry,
            budget-halt, and diversity-adjust edges.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2.5">
            {LAYERS.map((layer, index) => (
              <div key={layer.code} className="flex items-center gap-2.5">
                <div className="min-w-40 rounded-lg border border-niq-border border-t-4 border-t-niq-navy bg-white p-3 shadow-card">
                  <div className="text-xs font-bold text-niq-blue">{layer.code}</div>
                  <div className="font-bold text-niq-navy">{layer.name}</div>
                  <div className="text-xs text-niq-text-secondary">{layer.detail}</div>
                </div>
                {index < LAYERS.length - 1 && (
                  <span aria-hidden className="hidden text-niq-text-secondary lg:block">→</span>
                )}
              </div>
            ))}
          </div>
          <p className="mt-3 rounded-lg bg-niq-bg p-2.5 text-xs text-niq-text-secondary">
            <span className="font-bold text-niq-navy">Agent DAG:</span> SurveyDesigner →
            CohortSelector → TwinOrchestrator → Validator → CostAgent → CalibrationAgent →
            DiversityMonitor → AuditAgent
          </p>
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-3">
        <Link
          href="/survey"
          className="inline-flex h-10 items-center gap-2 rounded-lg bg-niq-navy px-5 text-sm font-semibold text-white transition-colors hover:bg-niq-blue"
        >
          <Target className="h-4 w-4" /> New Survey
        </Link>
        <Link
          href="/experiments"
          className="inline-flex h-10 items-center gap-2 rounded-lg border border-niq-border bg-niq-card px-5 text-sm font-semibold text-niq-navy transition-colors hover:border-niq-blue hover:text-niq-blue"
        >
          <FlaskConical className="h-4 w-4" /> Run Experiment
        </Link>
        <Link
          href="/results"
          className="inline-flex h-10 items-center gap-2 rounded-lg border border-niq-border bg-niq-card px-5 text-sm font-semibold text-niq-navy transition-colors hover:border-niq-blue hover:text-niq-blue"
        >
          <BarChart3 className="h-4 w-4" /> View Latest Results
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent survey runs</CardTitle>
          <CardDescription>Sample history — live runs appear here after execution.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Survey</TableHead>
                <TableHead>Respondents</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Cost</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {RECENT_RUNS.map((run) => (
                <TableRow key={run.survey}>
                  <TableCell className="font-semibold">{run.survey}</TableCell>
                  <TableCell>{formatNumber(run.respondents)}</TableCell>
                  <TableCell className="font-mono text-xs">{run.model}</TableCell>
                  <TableCell>{formatUsd(run.cost)}</TableCell>
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
