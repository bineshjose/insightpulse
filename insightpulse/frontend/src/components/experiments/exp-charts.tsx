"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EXP_COLORS, EXP_SERIES } from "@/lib/experiments-data";

const AXIS_TICK = { fill: "#6B7280", fontSize: 11 };

interface SeriesSpec {
  key: string;
  label: string;
  color?: string;
}

/** Grouped vertical bars, one group per row, one bar per metric. */
export function GroupedBars({
  data,
  xKey,
  series,
  height = 320,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  series: SeriesSpec[];
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ bottom: 24 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis dataKey={xKey} tick={AXIS_TICK} interval={0} angle={-16} textAnchor="end" />
        <YAxis tick={AXIS_TICK} />
        <Tooltip />
        <Legend verticalAlign="top" />
        {series.map((spec, index) => (
          <Bar
            key={spec.key}
            dataKey={spec.key}
            name={spec.label}
            fill={spec.color ?? EXP_SERIES[index % EXP_SERIES.length]}
            radius={[3, 3, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Single-metric bars with the selected row in green. */
export function SelectableBars({
  data,
  xKey,
  yKey,
  yLabel,
  height = 300,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  yLabel: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ bottom: 24 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis dataKey={xKey} tick={AXIS_TICK} interval={0} angle={-16} textAnchor="end" />
        <YAxis tick={AXIS_TICK} label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey={yKey} name={yLabel} radius={[3, 3, 0, 0]}>
          {data.map((row, index) => (
            <Cell key={index} fill={row.selected ? EXP_COLORS.green : EXP_COLORS.blue} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Multi-metric line chart over a shared x-axis, optional right axis. */
export function MultiLine({
  data,
  xKey,
  series,
  rightAxisKeys = [],
  xLabel,
  selectedX,
  height = 320,
  logX = false,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  series: SeriesSpec[];
  /** Keys plotted against a secondary right-hand axis. */
  rightAxisKeys?: string[];
  xLabel?: string;
  /** Draws a dashed green reference line at the selected x value. */
  selectedX?: number;
  height?: number;
  logX?: boolean;
}) {
  const hasRight = rightAxisKeys.length > 0;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ bottom: 20, right: hasRight ? 8 : 24 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis
          dataKey={xKey}
          type="number"
          scale={logX ? "log" : "auto"}
          domain={["dataMin", "dataMax"]}
          tick={AXIS_TICK}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -8, fill: "#6B7280", fontSize: 12 } : undefined}
        />
        <YAxis yAxisId="left" tick={AXIS_TICK} />
        {hasRight && <YAxis yAxisId="right" orientation="right" tick={AXIS_TICK} />}
        <Tooltip />
        <Legend verticalAlign="top" />
        {selectedX !== undefined && (
          <ReferenceLine
            yAxisId="left"
            x={selectedX}
            stroke={EXP_COLORS.green}
            strokeDasharray="6 3"
            label={{ value: "selected", fill: EXP_COLORS.green, fontSize: 11 }}
          />
        )}
        {series.map((spec, index) => (
          <Line
            key={spec.key}
            yAxisId={rightAxisKeys.includes(spec.key) ? "right" : "left"}
            dataKey={spec.key}
            name={spec.label}
            stroke={spec.color ?? EXP_SERIES[index % EXP_SERIES.length]}
            strokeWidth={2.5}
            dot={{ r: 3.5 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Sinkhorn convergence: marginal violation per iteration, log-scale y. */
export function LogConvergence({
  curves,
  selected,
  height = 340,
}: {
  curves: Record<string, { iterations: number[]; marginal_violation: number[] }>;
  selected: string;
  height?: number;
}) {
  const iterations = new Set<number>();
  Object.values(curves).forEach((curve) => curve.iterations.forEach((i) => iterations.add(i)));
  const data = [...iterations].sort((a, b) => a - b).map((iteration) => {
    const row: Record<string, number> = { iteration };
    Object.entries(curves).forEach(([name, curve]) => {
      const index = curve.iterations.indexOf(iteration);
      const value = index >= 0 ? curve.marginal_violation[index] : undefined;
      if (value !== undefined) row[name] = value;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ bottom: 20 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis
          dataKey="iteration"
          type="number"
          tick={AXIS_TICK}
          label={{ value: "Sinkhorn iteration", position: "insideBottom", offset: -8, fill: "#6B7280", fontSize: 12 }}
        />
        <YAxis
          scale="log"
          domain={[1e-7, 1]}
          tick={AXIS_TICK}
          tickFormatter={(value: number) => value.toExponential(0)}
          label={{ value: "Marginal violation (log)", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
        />
        <Tooltip formatter={(value: number) => value.toExponential(2)} />
        <Legend verticalAlign="top" />
        <ReferenceLine y={1e-6} stroke={EXP_COLORS.red} strokeDasharray="4 3" label={{ value: "tolerance 1e-6", fill: EXP_COLORS.red, fontSize: 11 }} />
        {Object.keys(curves).map((name, index) => (
          <Line
            key={name}
            dataKey={name}
            name={name.replace("eps_", "ε=") + (name === selected ? " (selected)" : "")}
            stroke={name === selected ? EXP_COLORS.green : EXP_SERIES[index % EXP_SERIES.length]}
            strokeWidth={name === selected ? 3.5 : 1.8}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Normalised radar over 5 quality axes (best model = 1.0 per axis). */
export function ModelRadar({
  entries,
  axes,
  height = 380,
}: {
  entries: Record<string, unknown>[];
  /** [key, label]; a `-` prefix marks lower-is-better (inverted). */
  axes: [string, string][];
  height?: number;
}) {
  const rows = axes.map(([key, label]) => {
    const rawKey = key.replace(/^-/, "");
    const values = entries.map((entry) => Number(entry[rawKey] ?? 0));
    const row: Record<string, unknown> = { axis: label };
    if (key.startsWith("-")) {
      const positives = values.filter((v) => v > 0);
      const best = positives.length > 0 ? Math.min(...positives) : 1;
      entries.forEach((entry, index) => {
        const value = values[index] ?? 0;
        row[String(entry.model)] = value > 0 ? Number((best / value).toFixed(3)) : 1;
      });
    } else {
      const best = Math.max(...values) || 1;
      entries.forEach((entry, index) => {
        row[String(entry.model)] = Number(((values[index] ?? 0) / best).toFixed(3));
      });
    }
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={rows}>
        <PolarGrid stroke={EXP_COLORS.grid} />
        <PolarAngleAxis dataKey="axis" tick={{ fill: "#374151", fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        {entries.map((entry, index) => (
          <Radar
            key={String(entry.model)}
            dataKey={String(entry.model)}
            name={String(entry.model)}
            stroke={EXP_SERIES[index % EXP_SERIES.length]}
            fill={EXP_SERIES[index % EXP_SERIES.length]}
            fillOpacity={0.12}
          />
        ))}
        <Legend verticalAlign="bottom" />
        <Tooltip />
      </RadarChart>
    </ResponsiveContainer>
  );
}

/** Cost-vs-quality scatter with model labels. */
export function CostQualityScatter({
  entries,
  height = 380,
}: {
  entries: { model: string; cost_usd: number; consistency: number }[];
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ bottom: 24, left: 8, right: 24, top: 20 }}>
        <CartesianGrid stroke={EXP_COLORS.grid} />
        <XAxis
          dataKey="cost_usd"
          type="number"
          name="Cost (USD)"
          tick={AXIS_TICK}
          label={{ value: "Cost per run (USD)", position: "insideBottom", offset: -10, fill: "#6B7280", fontSize: 12 }}
        />
        <YAxis
          dataKey="consistency"
          type="number"
          domain={[95, 100]}
          name="Consistency (%)"
          tick={AXIS_TICK}
          label={{ value: "Consistency (%)", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
        />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
        <Scatter data={entries} name="Models">
          {entries.map((entry, index) => (
            <Cell key={entry.model} fill={EXP_SERIES[index % EXP_SERIES.length]} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Ablation waterfall: JS deviation from the full-pipeline baseline. */
export function AblationWaterfall({
  rows,
  height = 360,
}: {
  rows: { config: string; js: number }[];
  height?: number;
}) {
  const baseline = rows[0]?.js ?? 0;
  const data = rows.slice(1).map((row) => {
    const delta = row.js - baseline;
    return {
      name: row.config.replace("Without ", "− "),
      base: Math.min(row.js, baseline),
      delta: Math.abs(delta),
      worse: delta > 0,
      js: row.js,
    };
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ bottom: 48 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis dataKey="name" tick={AXIS_TICK} interval={0} angle={-20} textAnchor="end" />
        <YAxis
          tick={AXIS_TICK}
          label={{ value: "JS divergence", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
        />
        <Tooltip
          formatter={(value: number, name: string) => (name === "delta" ? value.toFixed(3) : null)}
          labelFormatter={(label: string, payload) =>
            payload?.[0] ? `${label}: JS ${(payload[0].payload as { js: number }).js}` : label
          }
        />
        <ReferenceLine
          y={baseline}
          stroke={EXP_COLORS.navy}
          strokeDasharray="6 3"
          label={{ value: `full pipeline ${baseline}`, fill: EXP_COLORS.navy, fontSize: 11 }}
        />
        <Bar dataKey="base" stackId="wf" fill="transparent" />
        <Bar dataKey="delta" stackId="wf" radius={[3, 3, 0, 0]}>
          {data.map((row) => (
            <Cell key={row.name} fill={row.worse ? EXP_COLORS.red : EXP_COLORS.green} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Drift timeline with the alert threshold reference line. */
export function DriftTimeline({
  points,
  threshold,
  height = 320,
}: {
  points: { month: string; js_drift: number; annotation: string | null }[];
  threshold: number;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={points} margin={{ bottom: 20, top: 12 }}>
        <CartesianGrid vertical={false} stroke={EXP_COLORS.grid} />
        <XAxis dataKey="month" tick={AXIS_TICK} interval={0} angle={-16} textAnchor="end" />
        <YAxis
          domain={[0, threshold * 1.4]}
          tick={AXIS_TICK}
          label={{ value: "JS divergence", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
        />
        <Tooltip
          formatter={(value: number) => value.toFixed(3)}
          labelFormatter={(label: string, payload) => {
            const note = payload?.[0]
              ? (payload[0].payload as { annotation: string | null }).annotation
              : null;
            return note ? `${label} — ${note}` : label;
          }}
        />
        <ReferenceLine
          y={threshold}
          stroke={EXP_COLORS.red}
          strokeDasharray="6 3"
          label={{ value: `alert threshold ${threshold}`, fill: EXP_COLORS.red, fontSize: 11 }}
        />
        <Line dataKey="js_drift" name="Monthly JS drift" stroke={EXP_COLORS.blue} strokeWidth={2.5} dot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal log-scale bars for heterogeneous timing magnitudes. */
export function TimingBars({
  rows,
  height = 420,
}: {
  rows: { component: string; time_seconds: number; time_display: string; category: string }[];
  height?: number;
}) {
  const data = [...rows].sort((a, b) => a.time_seconds - b.time_seconds);
  const color: Record<string, string> = {
    training: EXP_COLORS.navy,
    inference: EXP_COLORS.blue,
    e2e: EXP_COLORS.green,
  };
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 160, right: 48 }}>
        <CartesianGrid horizontal={false} stroke={EXP_COLORS.grid} />
        <XAxis
          type="number"
          scale="log"
          domain={[0.0005, 10000]}
          tick={AXIS_TICK}
          tickFormatter={(value: number) => (value >= 1 ? `${value}s` : `${value * 1000}ms`)}
        />
        <YAxis type="category" dataKey="component" width={160} tick={{ fill: "#374151", fontSize: 11 }} />
        <Tooltip formatter={(_value: number, _name, item) => (item.payload as { time_display: string }).time_display} />
        <Bar dataKey="time_seconds" name="Time" radius={[0, 3, 3, 0]}>
          {data.map((row) => (
            <Cell key={row.component} fill={color[row.category] ?? EXP_COLORS.grey} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
