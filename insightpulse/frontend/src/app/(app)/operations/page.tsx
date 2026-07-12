"use client";

import { CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Lock } from "lucide-react";
import { Fragment, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricsCard } from "@/components/charts/metrics-card";
import { ModelRadarChart } from "@/components/charts/radar-chart";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useAuth } from "@/lib/auth";
import { CHART_SERIES } from "@/lib/demo-data";
import {
  ALERT_HISTORY,
  ALERT_RULES,
  ALERT_STATUS,
  API_ENDPOINTS,
  ARCHITECTURE_LAYERS,
  CURL_EXAMPLE,
  DATA_SOURCES,
  DEPENDENCIES,
  DESIGN_PATTERNS,
  ENDPOINT_PERF,
  LATENCY_PERCENTILES,
  LIFECYCLE_POLICY,
  LINEAGE_FLOWS,
  MODEL_PERF,
  MODEL_REGISTRY,
  PERF_KPIS,
  PROJECT_STATS,
  SECURITY_CONFIG,
  SECURITY_EVENTS,
  SECURITY_KPIS,
  SECURITY_MONITORING_NOTE,
  SYSTEM_HEALTH,
  THROUGHPUT_7D,
  type StatusTone,
} from "@/lib/ops-data";
import { cn, formatNumber } from "@/lib/utils";

const DOT_TONE: Record<StatusTone, string> = {
  green: "bg-niq-green",
  blue: "bg-niq-blue",
  amber: "bg-niq-amber",
  red: "bg-niq-red",
};

/** Severity → badge variant. Red is reserved for CRITICAL only. */
const SEVERITY_VARIANT: Record<string, "red" | "amber" | "blue" | "outline"> = {
  CRITICAL: "red",
  WARNING: "amber",
  MEDIUM: "amber",
  LOW: "blue",
  INFO: "outline",
};

/**
 * Operations: platform-administrator console — system health, security,
 * performance, alerting, model registry, data lineage, and project metrics.
 */
export default function OperationsPage() {
  const { user } = useAuth();
  const [expandedSource, setExpandedSource] = useState<string | null>(null);

  if (user && user.role !== "Platform Administrator") {
    return (
      <Card>
        <CardContent className="flex items-center gap-3 py-8">
          <Lock className="h-6 w-6 text-niq-amber" />
          <div>
            <p className="font-bold text-niq-navy">Access Restricted</p>
            <p className="text-sm text-niq-text-secondary">
              Your role ({user.role}) does not include Operations access —
              contact your administrator.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Operations</h1>
        <p className="text-sm text-niq-text-secondary">
          System health, security posture, performance, alerting, model
          lifecycle, data lineage, and engineering metrics.
        </p>
      </div>

      {/* 1 · System health */}
      <h2 className="text-lg font-bold text-niq-navy">System health</h2>
      <div className="flex items-center gap-2 rounded-lg border border-niq-green/40 bg-niq-green/10 px-4 py-3 text-sm font-semibold text-niq-green">
        <CheckCircle2 className="h-4 w-4" /> All systems operational
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {SYSTEM_HEALTH.map((component) => (
          <div
            key={component.component}
            className="rounded-xl border border-niq-border bg-niq-card p-4 shadow-card"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-niq-text-secondary">
                {component.component}
              </span>
              <StatusDot tone={component.tone} />
            </div>
            <div className="mt-1 text-lg font-bold text-niq-text">{component.status}</div>
            <div className="text-sm text-niq-text-secondary">{component.detail}</div>
          </div>
        ))}
      </div>

      {/* 2 · Security dashboard */}
      <h2 className="text-lg font-bold text-niq-navy">Security dashboard</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {SECURITY_KPIS.map((kpi) => (
          <MetricsCard
            key={kpi.label}
            label={kpi.label}
            value={kpi.value}
            delta={kpi.note}
            status={kpi.status}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Security events</CardTitle>
            <CardDescription>{SECURITY_MONITORING_NOTE}</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Event Type</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Details</TableHead>
                  <TableHead>Action Taken</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {SECURITY_EVENTS.map((event) => (
                  <TableRow key={`${event.timestamp}-${event.eventType}`}>
                    <TableCell className="whitespace-nowrap text-niq-text-secondary">
                      {event.timestamp}
                    </TableCell>
                    <TableCell className="font-semibold">{event.eventType}</TableCell>
                    <TableCell>
                      <Badge variant={SEVERITY_VARIANT[event.severity]}>{event.severity}</Badge>
                    </TableCell>
                    <TableCell>{event.details}</TableCell>
                    <TableCell>{event.action}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Security configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {SECURITY_CONFIG.map((item) => (
              <div
                key={item.setting}
                className="flex justify-between border-b border-niq-border pb-2 last:border-0"
              >
                <span className="text-niq-text-secondary">{item.setting}</span>
                <span className="flex items-center gap-1.5 font-semibold text-niq-navy">
                  <CheckCircle2 className="h-3.5 w-3.5 text-niq-green" /> {item.value}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* 3 · Performance metrics */}
      <h2 className="text-lg font-bold text-niq-navy">Performance metrics</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {PERF_KPIS.map((kpi) => (
          <MetricsCard
            key={kpi.label}
            label={kpi.label}
            value={kpi.value}
            delta={kpi.note}
            status={kpi.status}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Latency percentiles</CardTitle>
            <CardDescription>Per-response LLM generation latency (ms).</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={[...LATENCY_PERCENTILES]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="percentile" tick={{ fill: "#6B7280", fontSize: 12 }} />
                <YAxis
                  tick={{ fill: "#6B7280", fontSize: 11 }}
                  tickFormatter={(v: number) => `${v}ms`}
                />
                <Tooltip formatter={(v: number) => [`${formatNumber(v)}ms`, "Latency"]} />
                <Bar dataKey="ms" name="Latency" fill={CHART_SERIES[0]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Throughput — last 7 days</CardTitle>
            <CardDescription>Responses per minute, Jul 5 – Jul 11.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={[...THROUGHPUT_7D]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="day" tick={{ fill: "#6B7280", fontSize: 11 }} />
                <YAxis domain={[650, 850]} tick={{ fill: "#6B7280", fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [`${v} resp/min`, "Throughput"]} />
                <Line
                  dataKey="rpm"
                  name="Throughput"
                  stroke={CHART_SERIES[0]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Per-model performance</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Avg Latency</TableHead>
                  <TableHead>Avg Tokens</TableHead>
                  <TableHead>Cost/Response</TableHead>
                  <TableHead>Error Rate</TableHead>
                  <TableHead>Quality Score</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {MODEL_PERF.map((model) => (
                  <TableRow key={model.model}>
                    <TableCell className="font-mono text-xs">{model.model}</TableCell>
                    <TableCell>{model.avgLatency}</TableCell>
                    <TableCell>{model.avgTokens}</TableCell>
                    <TableCell>{model.costPerResponse}</TableCell>
                    <TableCell>{model.errorRate}</TableCell>
                    <TableCell className="font-semibold text-niq-navy">
                      {model.qualityScore}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>API endpoint performance</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {ENDPOINT_PERF.map((endpoint) => (
              <div
                key={endpoint.endpoint}
                className="flex items-baseline justify-between border-b border-niq-border pb-2 last:border-0"
              >
                <div>
                  <div className="font-mono text-xs font-semibold text-niq-navy">
                    {endpoint.endpoint}
                  </div>
                  <div className="text-xs text-niq-text-secondary">{endpoint.note}</div>
                </div>
                <span className="font-bold text-niq-text">avg {endpoint.avg}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* 4 · Alerts */}
      <h2 className="text-lg font-bold text-niq-navy">Active alerts &amp; history</h2>
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-wrap items-center gap-4 rounded-xl border border-niq-border bg-niq-card px-4 py-3 text-sm font-semibold shadow-card">
          {ALERT_STATUS.map((status, index) => (
            <Fragment key={status.severity}>
              {index > 0 && <span className="text-niq-border">·</span>}
              <span className="flex items-center gap-2">
                <StatusDot tone={status.tone} /> {status.count} {status.severity}
              </span>
            </Fragment>
          ))}
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-niq-green/40 bg-niq-green/10 px-4 py-3 text-sm font-semibold text-niq-green">
          <CheckCircle2 className="h-4 w-4" /> All systems healthy
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Alert history</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Alert</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Resolution</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ALERT_HISTORY.map((alert) => (
                  <TableRow key={alert.timestamp}>
                    <TableCell className="whitespace-nowrap text-niq-text-secondary">
                      {alert.timestamp}
                    </TableCell>
                    <TableCell>{alert.message}</TableCell>
                    <TableCell>
                      <Badge variant={SEVERITY_VARIANT[alert.severity]}>{alert.severity}</Badge>
                    </TableCell>
                    <TableCell>{alert.resolution}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Alert rules</CardTitle>
            <CardDescription>
              Nine rules evaluated after every run and on a 60s metrics loop.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Rule</TableHead>
                  <TableHead>Threshold</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Cooldown</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ALERT_RULES.map((rule) => (
                  <TableRow key={rule.rule}>
                    <TableCell className="font-semibold">{rule.rule}</TableCell>
                    <TableCell>{rule.threshold}</TableCell>
                    <TableCell>
                      <Badge variant={SEVERITY_VARIANT[rule.severity]}>{rule.severity}</Badge>
                    </TableCell>
                    <TableCell>{rule.cooldown}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* 5 · Model registry */}
      <h2 className="text-lg font-bold text-niq-navy">Model registry</h2>
      <Card>
        <CardHeader>
          <CardTitle>Registered models</CardTitle>
          <CardDescription>Lifecycle policy: {LIFECYCLE_POLICY}</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model ID</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Registered</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead>Runs</TableHead>
                <TableHead>Avg Quality</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {MODEL_REGISTRY.map((model) => (
                <TableRow key={model.modelId}>
                  <TableCell className="font-mono text-xs">{model.modelId}</TableCell>
                  <TableCell>{model.provider}</TableCell>
                  <TableCell>{model.version}</TableCell>
                  <TableCell>
                    <span className="flex items-center gap-2 font-semibold text-niq-green">
                      <StatusDot tone="green" /> {model.status}
                    </span>
                  </TableCell>
                  <TableCell>{model.registered}</TableCell>
                  <TableCell>{model.lastUsed}</TableCell>
                  <TableCell>{model.runs}</TableCell>
                  <TableCell className="font-semibold text-niq-navy">{model.avgQuality}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Model profile comparison</CardTitle>
          <CardDescription>
            Quality, speed, cost efficiency, low hallucination, and consistency —
            normalized so 100 = best observed value.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ModelRadarChart />
        </CardContent>
      </Card>

      {/* 6 · Data lineage */}
      <h2 className="text-lg font-bold text-niq-navy">Data lineage</h2>
      <Card>
        <CardHeader>
          <CardTitle>Pipeline flow</CardTitle>
          <CardDescription>
            Benchmark and panel paths converge into the L3 → L5 generation and
            insight layers.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {LINEAGE_FLOWS.map((flow, flowIndex) => (
            <div key={flow.name}>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-niq-text-secondary">
                {flowIndex === 2 ? "Both paths feed" : flow.name}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {flow.nodes.map((node, nodeIndex) => (
                  <div key={node} className="flex items-center gap-2">
                    <div
                      className={cn(
                        "rounded-lg border border-niq-border bg-white px-3 py-2 text-sm font-semibold shadow-card",
                        flowIndex === 2
                          ? "border-t-4 border-t-niq-navy text-niq-navy"
                          : "border-t-4 border-t-niq-blue text-niq-text",
                      )}
                    >
                      {node}
                    </div>
                    {nodeIndex < flow.nodes.length - 1 && (
                      <span aria-hidden className="text-niq-text-secondary">→</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data source status</CardTitle>
          <CardDescription>Click a row to expand schema and quality details.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8" aria-label="Expand" />
                <TableHead>Source</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Last Refreshed</TableHead>
                <TableHead>Records</TableHead>
                <TableHead>Quality</TableHead>
                <TableHead>Pipeline</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {DATA_SOURCES.map((source) => {
                const expanded = expandedSource === source.source;
                return (
                  <Fragment key={source.source}>
                    <TableRow
                      className="cursor-pointer hover:bg-niq-bg"
                      aria-expanded={expanded}
                      onClick={() =>
                        setExpandedSource(expanded ? null : source.source)
                      }
                    >
                      <TableCell className="text-niq-text-secondary">
                        {expanded ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                      </TableCell>
                      <TableCell className="font-semibold">{source.source}</TableCell>
                      <TableCell>{source.type}</TableCell>
                      <TableCell>{source.lastRefreshed}</TableCell>
                      <TableCell>{source.records}</TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5 font-semibold text-niq-green">
                          <CheckCircle2 className="h-3.5 w-3.5" /> {source.quality}
                        </span>
                      </TableCell>
                      <TableCell className="font-mono text-xs">{source.pipeline}</TableCell>
                    </TableRow>
                    {expanded && (
                      <TableRow>
                        <TableCell colSpan={7} className="bg-niq-bg">
                          <div className="grid grid-cols-1 gap-4 py-1 md:grid-cols-2">
                            <div>
                              <div className="mb-1.5 text-xs font-bold uppercase tracking-wider text-niq-text-secondary">
                                Schema
                              </div>
                              <div className="space-y-1">
                                {source.schema.map((column) => (
                                  <div key={column.field} className="flex gap-2 text-xs">
                                    <code className="font-mono font-semibold text-niq-navy">
                                      {column.field}
                                    </code>
                                    <span className="text-niq-text-secondary">{column.dtype}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div>
                              <div className="mb-1.5 text-xs font-bold uppercase tracking-wider text-niq-text-secondary">
                                Quality details
                              </div>
                              <ul className="space-y-1">
                                {source.qualityDetails.map((detail) => (
                                  <li key={detail} className="flex items-start gap-1.5 text-xs">
                                    <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-niq-green" />
                                    {detail}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* 7 · Project & code metrics */}
      <h2 className="text-lg font-bold text-niq-navy">Project &amp; code metrics</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {PROJECT_STATS.map((stat) => (
          <MetricsCard
            key={stat.label}
            label={stat.label}
            value={stat.value}
            delta={stat.note}
            status="info"
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Design patterns</CardTitle>
            <CardDescription>Applied across the backend codebase.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {DESIGN_PATTERNS.map((pattern) => (
              <span
                key={pattern}
                className="rounded-full border border-niq-border bg-niq-bg px-3 py-1 text-xs font-semibold text-niq-navy"
              >
                {pattern}
              </span>
            ))}
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Dependency summary</CardTitle>
            <CardDescription>Top 15 backend packages (pyproject.toml).</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Package</TableHead>
                  <TableHead>Version</TableHead>
                  <TableHead>Purpose</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {DEPENDENCIES.map((dep) => (
                  <TableRow key={dep.pkg}>
                    <TableCell className="font-mono text-xs font-semibold">{dep.pkg}</TableCell>
                    <TableCell className="font-mono text-xs">{dep.version}</TableCell>
                    <TableCell>{dep.purpose}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Architecture overview</CardTitle>
          <CardDescription>
            Five layers orchestrated by an 8-agent LangGraph DAG.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2.5">
            {ARCHITECTURE_LAYERS.map((layer, index) => (
              <div key={layer.code} className="flex items-center gap-2.5">
                <div className="min-w-40 rounded-lg border border-niq-border border-t-4 border-t-niq-navy bg-white p-3 shadow-card">
                  <div className="text-xs font-bold text-niq-blue">{layer.code}</div>
                  <div className="font-bold text-niq-navy">{layer.name}</div>
                  <div className="text-xs text-niq-text-secondary">{layer.detail}</div>
                </div>
                {index < ARCHITECTURE_LAYERS.length - 1 && (
                  <span aria-hidden className="hidden text-niq-text-secondary lg:block">→</span>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="group rounded-xl border border-niq-border bg-niq-card p-5 shadow-card transition-colors hover:border-niq-blue"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-niq-navy group-hover:text-niq-blue">
              API Documentation (Swagger)
            </span>
            <ExternalLink className="h-4 w-4 text-niq-text-secondary group-hover:text-niq-blue" />
          </div>
          <p className="mt-1 text-sm text-niq-text-secondary">
            Interactive OpenAPI reference for every endpoint — opens in a new tab.
          </p>
        </a>
        <a
          href="http://localhost:8000/metrics"
          target="_blank"
          rel="noreferrer"
          className="group rounded-xl border border-niq-border bg-niq-card p-5 shadow-card transition-colors hover:border-niq-blue"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-niq-navy group-hover:text-niq-blue">
              Prometheus Metrics
            </span>
            <ExternalLink className="h-4 w-4 text-niq-text-secondary group-hover:text-niq-blue" />
          </div>
          <p className="mt-1 text-sm text-niq-text-secondary">
            Raw scrape target with latency, throughput, and error counters.
          </p>
        </a>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Endpoint inventory</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Method</TableHead>
                <TableHead>Path</TableHead>
                <TableHead>Description</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {API_ENDPOINTS.map((endpoint) => (
                <TableRow key={`${endpoint.method}-${endpoint.path}`}>
                  <TableCell>
                    <Badge variant={endpoint.method === "POST" ? "blue" : "navy"}>
                      {endpoint.method}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs">{endpoint.path}</TableCell>
                  <TableCell>{endpoint.description}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div>
            <div className="mb-1.5 text-xs font-bold uppercase tracking-wider text-niq-text-secondary">
              Example request
            </div>
            <pre className="overflow-x-auto rounded-lg bg-niq-navy p-4 text-xs leading-relaxed text-white">
              {CURL_EXAMPLE}
            </pre>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function StatusDot({ tone }: { tone: StatusTone }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block h-2.5 w-2.5 shrink-0 rounded-full", DOT_TONE[tone])}
    />
  );
}
