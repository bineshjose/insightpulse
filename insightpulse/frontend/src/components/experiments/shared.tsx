"use client";

import type { ReactNode } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

/** Humanize a snake_case column key for table headers. */
export function humanize(key: string): string {
  const LABELS: Record<string, string> = {
    js: "JS divergence",
    js_downstream: "Downstream JS",
    js_precal: "JS (pre-calibration)",
    js_infit: "JS (in-fit)",
    wasserstein: "Wasserstein",
    hallucination: "Hallucination (%)",
    consistency: "Consistency (%)",
    entropy: "Entropy (bits)",
    coupling: "Coupling",
    coupling_entropy: "Coupling entropy",
    cosine: "Cosine similarity",
    cv_std: "CV std",
    silhouette: "Silhouette",
    latency_ms: "Latency (ms)",
    latency_sec: "Latency (s)",
    cost_usd: "Cost (USD)",
    train_hours: "Train time (h)",
    gpu_required: "GPU",
    parse_rate: "Parse rate (%)",
    params_millions: "Params (M)",
    training_min: "Training (min)",
    fit_time_sec: "Fit time (s)",
    index_mb: "Index size (MB)",
    sequences_per_household: "Sequences / household",
    effective_categories: "Effective categories",
    max_group_deviation_pp: "Max group deviation (pp)",
    js_penalty: "JS penalty",
    affected_group: "Affected group",
    tokens_per_response: "Tokens / response",
    contradiction_rate: "Contradictions (%)",
    spearman: "Spearman ρ",
    time_display: "Time",
    hardware_note: "Hardware",
    production_ready: "Production ready",
    rate_pct: "Rate (%)",
  };
  return LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

interface ResultsTableProps {
  rows: Record<string, unknown>[];
  /** Columns to hide (the `selected` flag is always hidden). */
  hide?: string[];
  /** Predicate deciding which rows get the green highlight. */
  isHighlighted?: (row: Record<string, unknown>) => boolean;
}

/** Comparison table with the selected/winning row highlighted in green. */
export function ResultsTable({ rows, hide = [], isHighlighted }: ResultsTableProps) {
  const first = rows[0];
  if (!first) return null;
  const hidden = new Set(["selected", ...hide]);
  const columns = Object.keys(first).filter((key) => !hidden.has(key));
  const highlighted = isHighlighted ?? ((row) => row.selected === true);

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column}>{humanize(column)}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, index) => {
            const active = highlighted(row);
            return (
              <TableRow
                key={index}
                className={cn(active && "bg-niq-green/10 font-semibold text-green-900")}
              >
                {columns.map((column) => (
                  <TableCell key={column}>{formatCell(row[column])}</TableCell>
                ))}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

/** Finding text block rendered under each chart. */
export function Finding({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-md border-l-4 border-niq-blue bg-niq-blue/5 px-4 py-2 text-sm text-niq-text">
      <span className="font-bold">Finding.</span> {children}
    </div>
  );
}

/** Sub-section wrapper: title + content, mirrors the analyst dashboard. */
export function SubSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3 border-b border-niq-border pb-6 last:border-b-0">
      <h3 className="text-base font-bold text-niq-navy">{title}</h3>
      {children}
    </section>
  );
}

/** Green callout badge for headline takeaways. */
export function Callout({ children, tone = "green" }: { children: ReactNode; tone?: "green" | "red" }) {
  return (
    <div
      className={cn(
        "rounded-md border px-4 py-2.5 text-sm font-medium",
        tone === "green"
          ? "border-niq-green bg-niq-green/10 text-green-900"
          : "border-niq-red bg-niq-red/10 text-red-900",
      )}
    >
      {children}
    </div>
  );
}

/** Small KPI card used for before/after and turnaround comparisons. */
export function StatCard({
  label,
  value,
  sub,
  tone = "info",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "good" | "warn" | "info";
}) {
  const border = { good: "border-l-niq-green", warn: "border-l-niq-amber", info: "border-l-niq-blue" }[tone];
  return (
    <div className={cn("rounded-md border border-niq-border border-l-4 bg-niq-card p-3", border)}>
      <div className="text-[0.7rem] font-bold uppercase tracking-wide text-niq-text-secondary">{label}</div>
      <div className="mt-1 text-xl font-bold text-niq-navy">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-niq-text-secondary">{sub}</div>}
    </div>
  );
}
