"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

const STAGES = [
  "SurveyDesigner — structuring questions",
  "CohortSelector — sampling panelists",
  "TwinOrchestrator — generating responses",
  "Validator — quality checks",
  "CalibrationAgent — Sinkhorn alignment",
  "InsightEngine — compiling analytics",
] as const;

/**
 * Execution progress with agent-stage indicators. The backend endpoint is
 * synchronous, so stage advancement is time-based (paced to typical stage
 * durations) and snaps to complete when the response arrives.
 */
export function RunProgress({ done }: { done: boolean }) {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    if (done) {
      setStage(STAGES.length);
      return;
    }
    const timer = setInterval(
      () => setStage((current) => Math.min(current + 1, STAGES.length - 1)),
      1800,
    );
    return () => clearInterval(timer);
  }, [done]);

  const percent = done ? 100 : Math.min(92, ((stage + 1) / STAGES.length) * 100);

  return (
    <div className="space-y-3">
      <Progress value={percent} />
      <ol className="space-y-1.5">
        {STAGES.map((label, index) => {
          const complete = done || index < stage;
          const active = !done && index === stage;
          return (
            <li
              key={label}
              className={cn(
                "flex items-center gap-2 text-sm",
                complete ? "text-niq-green" : active ? "text-niq-navy" : "text-niq-text-secondary/60",
              )}
            >
              {complete ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : active ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <span className="inline-block h-4 w-4 rounded-full border border-niq-border" />
              )}
              {label}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
