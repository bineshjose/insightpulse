"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SERIES_COLORS } from "@/lib/demo-data";
import type { SurveyResult } from "@/lib/types";

interface DistributionChartProps {
  result: SurveyResult;
  /** Show the calibrated series (hidden when calibration was off). */
  showCalibrated?: boolean;
}

/**
 * Grouped bar chart of one question's response distribution:
 * raw synthetic vs BDCL-calibrated (percentages, fixed series colors).
 */
export function DistributionChart({ result, showCalibrated = true }: DistributionChartProps) {
  const data = result.options.map((option, index) => ({
    option: option.split(" ")[0], // short tick labels; full text in tooltip
    fullOption: option,
    raw: result.distribution[index]?.percentage ?? 0,
    calibrated: result.calibrated_distribution
      ? Number(((result.calibrated_distribution[index] ?? 0) * 100).toFixed(1))
      : 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} barGap={2}>
        <CartesianGrid vertical={false} stroke="#E5E7EB" />
        <XAxis dataKey="option" tick={{ fill: "#6B7280", fontSize: 12 }} />
        <YAxis
          tick={{ fill: "#6B7280", fontSize: 12 }}
          label={{
            value: "% of respondents",
            angle: -90,
            position: "insideLeft",
            fill: "#6B7280",
            fontSize: 12,
          }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [`${value}%`, name]}
          labelFormatter={(label: string, payload) =>
            payload?.[0]?.payload?.fullOption ?? label
          }
        />
        <Legend />
        <Bar dataKey="raw" name="Synthetic (raw)" fill={SERIES_COLORS.raw} radius={[4, 4, 0, 0]} />
        {showCalibrated && result.calibrated_distribution && (
          <Bar
            dataKey="calibrated"
            name="Calibrated"
            fill={SERIES_COLORS.calibrated}
            radius={[4, 4, 0, 0]}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}
