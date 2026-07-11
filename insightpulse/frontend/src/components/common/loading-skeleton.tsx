import { cn } from "@/lib/utils";

/** Pulsing skeleton block; compose to match the card being loaded. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-lg bg-niq-border/70", className)} />;
}

/** Skeleton matching the KPI card row (4 cards). */
export function KpiRowSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {Array.from({ length: 4 }, (_, index) => (
        <Skeleton key={index} className="h-24" />
      ))}
    </div>
  );
}

/** Skeleton matching a full-width chart card. */
export function ChartSkeleton() {
  return <Skeleton className="h-72 w-full" />;
}
