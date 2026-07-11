import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Status = "good" | "warn" | "bad" | "info";

const BORDER: Record<Status, string> = {
  good: "border-l-niq-green",
  warn: "border-l-niq-amber",
  bad: "border-l-niq-red",
  info: "border-l-niq-blue",
};

const DELTA_TEXT: Record<Status, string> = {
  good: "text-niq-green",
  warn: "text-niq-amber",
  bad: "text-niq-red",
  info: "text-niq-text-secondary",
};

export interface MetricsCardProps {
  label: string;
  value: string;
  /** Secondary line: trend, target note, context. */
  delta?: string;
  /** Colors the left border and the delta line. */
  status?: Status;
  icon?: LucideIcon;
}

/** KPI card: icon, value, label, delta — colored left border by status. */
export function MetricsCard({
  label,
  value,
  delta,
  status = "info",
  icon: Icon,
}: MetricsCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-niq-border border-l-4 bg-niq-card p-4 shadow-card",
        BORDER[status],
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-niq-text-secondary">
          {label}
        </span>
        {Icon && <Icon aria-hidden className="h-4 w-4 text-niq-text-secondary" />}
      </div>
      <div className="mt-1 text-2xl font-bold text-niq-text">{value}</div>
      {delta && (
        <div className={cn("mt-0.5 text-sm font-semibold", DELTA_TEXT[status])}>{delta}</div>
      )}
    </div>
  );
}
