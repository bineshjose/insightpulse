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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CHART_SERIES } from "@/lib/demo-data";

/**
 * Tabbed demographic slicing: answer share per group across
 * age / income / region / cluster dimensions.
 */

const DIMENSIONS = ["Age", "Income", "Region", "Cluster"] as const;

/** Deterministic shares per dimension (mirrors the demo panel). */
const BREAKDOWNS: Record<(typeof DIMENSIONS)[number], { group: string; low: number; mid: number; high: number }[]> = {
  Age: [
    { group: "18-24", low: 32, mid: 41, high: 27 },
    { group: "25-34", low: 24, mid: 40, high: 36 },
    { group: "35-44", low: 26, mid: 42, high: 32 },
    { group: "45-54", low: 30, mid: 43, high: 27 },
    { group: "55-64", low: 35, mid: 41, high: 24 },
    { group: "65+", low: 41, mid: 38, high: 21 },
  ],
  Income: [
    { group: "Low", low: 45, mid: 38, high: 17 },
    { group: "Lower-mid", low: 38, mid: 41, high: 21 },
    { group: "Middle", low: 29, mid: 43, high: 28 },
    { group: "Upper-mid", low: 20, mid: 42, high: 38 },
    { group: "High", low: 13, mid: 38, high: 49 },
  ],
  Region: [
    { group: "Northeast", low: 27, mid: 42, high: 31 },
    { group: "Midwest", low: 31, mid: 41, high: 28 },
    { group: "South", low: 33, mid: 40, high: 27 },
    { group: "West", low: 24, mid: 41, high: 35 },
  ],
  Cluster: [
    { group: "Value seeker", low: 48, mid: 37, high: 15 },
    { group: "Premium loyalist", low: 12, mid: 36, high: 52 },
    { group: "Impulse buyer", low: 33, mid: 44, high: 23 },
    { group: "Health conscious", low: 9, mid: 33, high: 58 },
    { group: "Bulk planner", low: 37, mid: 42, high: 21 },
  ],
};

export function DemographicBreakdown() {
  return (
    <Tabs defaultValue="Age">
      <TabsList>
        {DIMENSIONS.map((dimension) => (
          <TabsTrigger key={dimension} value={dimension}>
            {dimension}
          </TabsTrigger>
        ))}
      </TabsList>
      {DIMENSIONS.map((dimension) => (
        <TabsContent key={dimension} value={dimension}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={BREAKDOWNS[dimension]} barGap={2}>
              <CartesianGrid vertical={false} stroke="#E5E7EB" />
              <XAxis dataKey="group" tick={{ fill: "#6B7280", fontSize: 12 }} />
              <YAxis
                tick={{ fill: "#6B7280", fontSize: 12 }}
                label={{ value: "% of group", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
              />
              <Tooltip formatter={(value: number) => `${value}%`} />
              <Legend />
              <Bar dataKey="low" name="Low importance" fill={CHART_SERIES[0]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="mid" name="Moderate" fill={CHART_SERIES[1]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="high" name="High importance" fill={CHART_SERIES[2]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </TabsContent>
      ))}
    </Tabs>
  );
}
