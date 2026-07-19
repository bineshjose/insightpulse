"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { getHealth } from "@/lib/api";

export type OperationalMode = "demo" | "api" | "prod";

interface ModeSpec {
  label: string;
  variant: "outline" | "blue" | "green";
  className?: string;
}

const MODE_SPECS: Record<OperationalMode, ModeSpec> = {
  demo: {
    label: "Prod API - Offline",
    variant: "outline",
    className: "border-niq-navy text-niq-navy",
  },
  api: { label: "API Mode", variant: "blue" },
  prod: { label: "Production Mode", variant: "green" },
};

function normalizeMode(env: string | null | undefined): OperationalMode {
  const value = (env ?? "demo").toLowerCase();
  if (value === "api") return "api";
  if (value === "prod" || value === "production") return "prod";
  return "demo";
}

/** Resolve the backend's operational mode via /health (demo fallback). */
export function useMode(): OperationalMode {
  const [mode, setMode] = useState<OperationalMode>(
    normalizeMode(process.env.NEXT_PUBLIC_ENV),
  );

  useEffect(() => {
    getHealth()
      .then((health) => setMode(normalizeMode(health.env)))
      .catch(() => undefined);
  }, []);

  return mode;
}

/** Header status badge naming the operational mode. */
export function ModeBadge({ mode }: { mode: OperationalMode }) {
  const spec = MODE_SPECS[mode];
  return (
    <Badge variant={spec.variant} className={spec.className}>
      {spec.label}
    </Badge>
  );
}
