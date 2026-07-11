import { cn } from "@/lib/utils";

interface ProgressProps {
  /** Completion in [0, 100]. */
  value: number;
  className?: string;
  /** Bar color class (defaults to NIQ navy). */
  barClassName?: string;
}

/** Determinate progress bar (navy fill on hairline track). */
export function Progress({ value, className, barClassName }: ProgressProps) {
  const clamped = Math.min(100, Math.max(0, value));
  return (
    <div
      role="progressbar"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-niq-border", className)}
    >
      <div
        className={cn("h-full rounded-full bg-niq-navy transition-all", barClassName)}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}
