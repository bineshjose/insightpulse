"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CHART_SERIES, CONVERGENCE_BY_EPSILON } from "@/lib/demo-data";

interface ConvergenceChartProps {
  /** ε values to plot (keys of CONVERGENCE_BY_EPSILON). */
  epsilons: string[];
}

/**
 * Sinkhorn convergence curves: marginal error per iteration, one line per ε
 * (log-scale y — convergence rates span decades).
 */
export function ConvergenceChart({ epsilons }: ConvergenceChartProps) {
  // Merge the per-ε series into one row per iteration for recharts.
  const iterations = new Set<number>();
  epsilons.forEach((eps) =>
    CONVERGENCE_BY_EPSILON[eps]?.forEach((point) => iterations.add(point.iteration)),
  );
  const data = [...iterations]
    .sort((a, b) => a - b)
    .map((iteration) => {
      const row: Record<string, number> = { iteration };
      epsilons.forEach((eps) => {
        const point = CONVERGENCE_BY_EPSILON[eps]?.find((p) => p.iteration === iteration);
        if (point) row[`eps${eps}`] = point.error;
      });
      return row;
    });

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
        <CartesianGrid vertical={false} stroke="#E5E7EB" />
        <XAxis
          dataKey="iteration"
          type="number"
          tick={{ fill: "#6B7280", fontSize: 12 }}
          label={{ value: "Sinkhorn iteration", position: "insideBottom", offset: -4, fill: "#6B7280", fontSize: 12 }}
        />
        <YAxis
          scale="log"
          domain={["auto", "auto"]}
          tick={{ fill: "#6B7280", fontSize: 11 }}
          tickFormatter={(value: number) => value.toExponential(0)}
          label={{ value: "marginal error (log)", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
        />
        <Tooltip formatter={(value: number) => value.toExponential(2)} />
        <Legend />
        {epsilons.map((eps, index) => (
          <Line
            key={eps}
            dataKey={`eps${eps}`}
            name={`ε = ${eps}`}
            stroke={CHART_SERIES[index % CHART_SERIES.length]}
            strokeWidth={2}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
