"use client";

import { Lock } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricsCard } from "@/components/charts/metrics-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  AGE_DISTRIBUTION,
  ARCHETYPE_RADAR,
  CATEGORY_PENETRATION,
  CLUSTER_PROFILES,
  HOUSEHOLD_SIZE_DISTRIBUTION,
  INCOME_DISTRIBUTION,
  MONTHLY_VOLUME,
  PANEL_SUMMARY,
  PRICE_TIER_DISTRIBUTION,
  PROMOTION_BY_ARCHETYPE,
  PURCHASE_SUMMARY,
  QUALITY_GATES,
  REGION_DISTRIBUTION,
  SILHOUETTE_OVERALL,
} from "@/lib/eda-data";
import { formatNumber, formatPercent } from "@/lib/utils";

const ALLOWED_ROLES = ["Platform Administrator", "Read-Only Evaluator"];

const RADAR_DATA = ARCHETYPE_RADAR.dimensions.map((dimension, i) => ({
  dimension,
  ...Object.fromEntries(
    Object.entries(ARCHETYPE_RADAR.profiles).map(([name, values]) => [name, values[i]]),
  ),
}));

/** Data Explorer: panel composition, purchases, archetypes, data quality. */
export default function DataExplorerPage() {
  const { user } = useAuth();

  if (user && !ALLOWED_ROLES.includes(user.role)) {
    return (
      <Card>
        <CardContent className="flex items-center gap-3 py-8">
          <Lock className="h-6 w-6 text-niq-amber" />
          <div>
            <p className="font-bold text-niq-navy">Access Restricted</p>
            <p className="text-sm text-niq-text-secondary">
              Your role ({user.role}) does not include Data Explorer access —
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
        <h1 className="text-2xl font-bold text-niq-navy">Data Explorer</h1>
        <p className="text-sm text-niq-text-secondary">
          Panel composition, purchase behavior, behavioral archetypes, and data
          quality for the active panel.
        </p>
      </div>

      {/* 1 · Panel composition */}
      <h2 className="text-lg font-bold text-niq-navy">Panel composition</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricsCard label="Total households" value={formatNumber(PANEL_SUMMARY.totalHouseholds)} status="info" />
        <MetricsCard label="Total members" value={formatNumber(PANEL_SUMMARY.totalMembers)} delta="estimated" status="info" />
        <MetricsCard label="Avg household size" value={String(PANEL_SUMMARY.avgHouseholdSize)} status="info" />
        <MetricsCard label="Panel coverage" value={`${PANEL_SUMMARY.regionsCovered} regions`} status="info" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DistributionCard title="Age group distribution" data={AGE_DISTRIBUTION.map((d) => ({ group: d.group, share: d.share }))} />
        <DistributionCard title="Income group distribution" data={INCOME_DISTRIBUTION.map((d) => ({ group: d.group, share: d.share }))} />
        <DistributionCard title="Region distribution" data={REGION_DISTRIBUTION.map((d) => ({ group: d.group, share: d.share }))} />
        <Card>
          <CardHeader>
            <CardTitle>Household size</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={HOUSEHOLD_SIZE_DISTRIBUTION.map((d) => ({
                    name: d.group,
                    value: d.households,
                  }))}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={55}
                  outerRadius={85}
                  label={(entry) => entry.name}
                >
                  {HOUSEHOLD_SIZE_DISTRIBUTION.map((d, i) => (
                    <Cell key={d.group} fill={CHART_SERIES[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 2 · Purchase behavior */}
      <h2 className="text-lg font-bold text-niq-navy">Purchase behavior</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricsCard label="Total transactions" value={formatNumber(PURCHASE_SUMMARY.totalTransactions)} status="info" />
        <MetricsCard label="Avg basket value" value={`$${PURCHASE_SUMMARY.avgBasketValue.toFixed(2)}`} status="info" />
        <MetricsCard label="Avg unit price" value={`$${PURCHASE_SUMMARY.avgUnitPrice.toFixed(2)}`} status="info" />
        <MetricsCard label="Promotion rate" value={formatPercent(PURCHASE_SUMMARY.promotionRate)} status="info" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DistributionCard
          title="Category penetration"
          data={CATEGORY_PENETRATION.map((d) => ({ group: d.category, share: d.share }))}
        />
        <Card>
          <CardHeader>
            <CardTitle>Price tier distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={[...PRICE_TIER_DISTRIBUTION]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="tier" tick={{ fill: "#6B7280", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="transactions" name="Transactions" radius={[4, 4, 0, 0]}>
                  {PRICE_TIER_DISTRIBUTION.map((_, i) => (
                    <Cell key={i} fill={CHART_SERIES[i % CHART_SERIES.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Promotion response by archetype</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={[...PROMOTION_BY_ARCHETYPE]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="archetype" tick={{ fill: "#6B7280", fontSize: 10 }} />
                <YAxis tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fill: "#6B7280", fontSize: 11 }} />
                <Tooltip formatter={(v: number) => formatPercent(v)} />
                <Bar dataKey="share" name="Promotion share" fill={CHART_SERIES[1]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Monthly transaction volume</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={[...MONTHLY_VOLUME]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="month" tick={{ fill: "#6B7280", fontSize: 10 }} />
                <YAxis domain={[600, 1000]} tick={{ fill: "#6B7280", fontSize: 11 }} />
                <Tooltip />
                <Line dataKey="transactions" name="Transactions" stroke={CHART_SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 3 · Behavioral archetypes */}
      <h2 className="text-lg font-bold text-niq-navy">Behavioral archetypes</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {CLUSTER_PROFILES.map((cluster) => (
          <MetricsCard
            key={cluster.name}
            label={cluster.name}
            value={formatPercent(cluster.share)}
            delta={`silhouette ${cluster.silhouette.toFixed(2)}`}
            status="info"
          />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Archetype behavioral profiles</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={340}>
              <RadarChart data={RADAR_DATA} outerRadius={110}>
                <PolarGrid stroke="#E5E7EB" />
                <PolarAngleAxis dataKey="dimension" tick={{ fill: "#6B7280", fontSize: 10 }} />
                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                {Object.keys(ARCHETYPE_RADAR.profiles).map((name, i) => (
                  <Radar
                    key={name}
                    name={name}
                    dataKey={name}
                    stroke={CHART_SERIES[i % CHART_SERIES.length]}
                    fill={CHART_SERIES[i % CHART_SERIES.length]}
                    fillOpacity={0.06}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Cluster sizes — overall silhouette {SILHOUETTE_OVERALL}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={[...CLUSTER_PROFILES]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="name" tick={{ fill: "#6B7280", fontSize: 10 }} />
                <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="households" name="Households" radius={[4, 4, 0, 0]}>
                  {CLUSTER_PROFILES.map((_, i) => (
                    <Cell key={i} fill={CHART_SERIES[i % CHART_SERIES.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 4 · Data quality */}
      <h2 className="text-lg font-bold text-niq-navy">Data quality</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {QUALITY_GATES.map((gate) => (
          <MetricsCard
            key={gate.gate}
            label={gate.gate}
            value={gate.value}
            delta={gate.pass ? `✓ Passes (${gate.target})` : `✗ ${gate.target}`}
            status={gate.pass ? "good" : "bad"}
          />
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Cluster summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Archetype</TableHead>
                <TableHead>Households</TableHead>
                <TableHead>Share</TableHead>
                <TableHead>Silhouette</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {CLUSTER_PROFILES.map((cluster) => (
                <TableRow key={cluster.name}>
                  <TableCell className="font-semibold">{cluster.name}</TableCell>
                  <TableCell>{cluster.households}</TableCell>
                  <TableCell>{formatPercent(cluster.share)}</TableCell>
                  <TableCell>{cluster.silhouette.toFixed(2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function DistributionCard({
  title,
  data,
}: {
  title: string;
  data: { group: string; share: number }[];
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 34)}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid horizontal={false} stroke="#E5E7EB" />
            <XAxis
              type="number"
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tick={{ fill: "#6B7280", fontSize: 11 }}
            />
            <YAxis type="category" dataKey="group" width={100} tick={{ fill: "#6B7280", fontSize: 11 }} />
            <Tooltip formatter={(v: number) => formatPercent(v)} />
            <Bar dataKey="share" name="Share" fill={CHART_SERIES[0]} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
