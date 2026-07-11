"use client";

import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart as RechartsRadar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { CHART_SERIES, MODEL_COMPARISON } from "@/lib/demo-data";

/**
 * Multi-model comparison radar. Every axis is normalized so that 100 =
 * best observed value (higher is always better after normalization —
 * error-type metrics are inverted).
 */
export function ModelRadarChart() {
  const axes = [
    { key: "fidelity", label: "Distribution fidelity" },
    { key: "ordinal", label: "Ordinal accuracy" },
    { key: "trust", label: "Low hallucination" },
    { key: "consistency", label: "Consistency" },
    { key: "economy", label: "Cost efficiency" },
    { key: "speed", label: "Throughput" },
  ] as const;

  const minJs = Math.min(...MODEL_COMPARISON.map((m) => m.jsDivergence));
  const minWs = Math.min(...MODEL_COMPARISON.map((m) => m.wasserstein));
  const minHall = Math.min(...MODEL_COMPARISON.map((m) => m.hallucination));
  const maxThroughput = Math.max(...MODEL_COMPARISON.map((m) => m.throughput));

  const scores = MODEL_COMPARISON.map((m) => ({
    model: m.model,
    fidelity: Math.round((minJs / m.jsDivergence) * 100),
    ordinal: Math.round((minWs / m.wasserstein) * 100),
    trust: Math.round((minHall / m.hallucination) * 100),
    consistency: Math.round(m.consistency * 100),
    economy: m.costUsd === 0 ? 100 : Math.round((0.1 / m.costUsd) * 100),
    speed: Math.round((m.throughput / maxThroughput) * 100),
  }));

  const data = axes.map((axis) => {
    const row: Record<string, string | number> = { axis: axis.label };
    scores.forEach((score) => {
      row[score.model] = score[axis.key];
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={340}>
      <RechartsRadar data={data} outerRadius="72%">
        <PolarGrid stroke="#E5E7EB" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: "#6B7280", fontSize: 11 }} />
        <Tooltip />
        <Legend />
        {scores.map((score, index) => (
          <Radar
            key={score.model}
            name={score.model}
            dataKey={score.model}
            stroke={CHART_SERIES[index % CHART_SERIES.length]}
            fill={CHART_SERIES[index % CHART_SERIES.length]}
            fillOpacity={0.12}
            strokeWidth={2}
          />
        ))}
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
