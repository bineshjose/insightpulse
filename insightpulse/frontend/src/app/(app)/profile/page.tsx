"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Avatar } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { CHART_SERIES, SUPPORTED_MODELS } from "@/lib/demo-data";
import { useAuth } from "@/lib/auth";
import { formatNumber } from "@/lib/utils";
import type { SubscriptionTier } from "@/lib/types";

const TIER_VARIANT: Record<SubscriptionTier, "navy" | "blue" | "green"> = {
  Enterprise: "navy",
  Professional: "blue",
  Academic: "green",
};

const USAGE_TREND = [
  { week: "W1", surveys: 3, responses: 420 },
  { week: "W2", surveys: 5, responses: 780 },
  { week: "W3", surveys: 4, responses: 610 },
  { week: "W4", surveys: 7, responses: 1240 },
  { week: "W5", surveys: 6, responses: 980 },
  { week: "W6", surveys: 9, responses: 1660 },
] as const;

/** Profile: account card, editable preferences, usage charts. */
export default function ProfilePage() {
  const { user } = useAuth();
  const [editing, setEditing] = useState(false);
  const [displayName, setDisplayName] = useState(user?.name ?? "");
  const [title, setTitle] = useState(user?.title ?? "");
  const [region, setRegion] = useState(user?.regions[0] ?? "");
  const [defaultModel, setDefaultModel] = useState<string>(SUPPORTED_MODELS[0]);
  const [notifyEmail, setNotifyEmail] = useState(true);
  const [notifyCompletion, setNotifyCompletion] = useState(true);
  const [saved, setSaved] = useState(false);

  if (!user) return null;
  const creditsUsed = user.creditsTotal - user.creditsBalance;
  const callsUsed = user.apiCallsQuota - user.apiCallsRemaining;

  const save = () => {
    setEditing(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-5">
      <Card>
        <CardContent className="flex flex-wrap items-center gap-5 py-5">
          <Avatar initials={user.initials} tier={user.tier} className="h-20 w-20 text-2xl" />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-xl font-bold text-niq-text">{displayName}</h1>
              <Badge variant={TIER_VARIANT[user.tier]}>{user.tier}</Badge>
            </div>
            <p className="text-sm text-niq-text-secondary">{user.email}</p>
            <p className="text-sm text-niq-text">
              {user.role} · {title}
            </p>
            <p className="text-xs text-niq-text-secondary">
              {user.department} · Regions: {user.regions.join(", ")}
            </p>
          </div>
          <Button variant={editing ? "secondary" : "default"} onClick={() => setEditing((v) => !v)}>
            {editing ? "Cancel" : "Edit profile"}
          </Button>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Settings</CardTitle>
            <CardDescription>
              {editing ? "Editing — changes apply to this session." : "Toggle edit mode to change preferences."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="display-name">Display name</Label>
              <Input
                id="display-name"
                disabled={!editing}
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="profile-title">Title</Label>
              <Input
                id="profile-title"
                disabled={!editing}
                value={title}
                onChange={(event) => setTitle(event.target.value)}
              />
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <Label htmlFor="region">Preferred region</Label>
                <Select
                  id="region"
                  disabled={!editing}
                  options={user.regions.map((r) => ({ value: r, label: r }))}
                  value={region}
                  onChange={(event) => setRegion(event.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="default-model">Default LLM model</Label>
                <Select
                  id="default-model"
                  disabled={!editing}
                  options={SUPPORTED_MODELS.map((m) => ({ value: m, label: m }))}
                  value={defaultModel}
                  onChange={(event) => setDefaultModel(event.target.value)}
                />
              </div>
            </div>
            <div className="space-y-2.5 border-t border-niq-border pt-3">
              <div className="flex items-center justify-between text-sm">
                <span>Email alerts</span>
                <Switch checked={notifyEmail} onCheckedChange={setNotifyEmail} disabled={!editing} aria-label="Email alerts" />
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Survey completion notifications</span>
                <Switch checked={notifyCompletion} onCheckedChange={setNotifyCompletion} disabled={!editing} aria-label="Completion notifications" />
              </div>
            </div>
            {editing && (
              <Button className="w-full" onClick={save}>
                Save Changes
              </Button>
            )}
            {saved && (
              <p className="rounded-lg bg-niq-green/10 px-3 py-2 text-sm font-semibold text-niq-green">
                ✓ Profile settings saved
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Account & quota</CardTitle>
            <CardDescription>
              Created {user.created} · read-only, managed by your administrator.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div>
              <div className="mb-1 flex justify-between">
                <span className="text-niq-text-secondary">Credits</span>
                <span className="font-semibold">
                  {formatNumber(user.creditsBalance)} / {formatNumber(user.creditsTotal)} remaining
                </span>
              </div>
              <Progress value={(creditsUsed / user.creditsTotal) * 100} />
            </div>
            <div>
              <div className="mb-1 flex justify-between">
                <span className="text-niq-text-secondary">API calls this month</span>
                <span className="font-semibold">
                  {formatNumber(user.apiCallsRemaining)} / {formatNumber(user.apiCallsQuota)} remaining
                </span>
              </div>
              <Progress value={Math.max(2, (callsUsed / user.apiCallsQuota) * 100)} />
            </div>
            <div className="flex justify-between border-t border-niq-border pt-3">
              <span className="text-niq-text-secondary">Max cohort size</span>
              <span className="font-semibold">{formatNumber(user.maxCohortSize)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-niq-text-secondary">Permissions</span>
              <span className="text-right font-mono text-xs">{user.permissions.join(" · ")}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Usage — last 6 weeks</CardTitle>
          <CardDescription>Surveys run and responses generated per week.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div>
            <p className="mb-1 text-xs font-semibold uppercase text-niq-text-secondary">Surveys</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[...USAGE_TREND]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="week" tick={{ fill: "#6B7280", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6B7280", fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="surveys" name="Surveys" fill={CHART_SERIES[0]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <p className="mb-1 text-xs font-semibold uppercase text-niq-text-secondary">Responses</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[...USAGE_TREND]}>
                <CartesianGrid vertical={false} stroke="#E5E7EB" />
                <XAxis dataKey="week" tick={{ fill: "#6B7280", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6B7280", fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="responses" name="Responses" fill={CHART_SERIES[1]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
