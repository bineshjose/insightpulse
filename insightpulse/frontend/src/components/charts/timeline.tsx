import type { AgentTraceEntry } from "@/lib/types";

interface TimelineProps {
  trace: AgentTraceEntry[];
}

/**
 * Vertical timeline of the agent execution trace, with duration bars
 * scaled to the slowest agent.
 */
export function AgentTimeline({ trace }: TimelineProps) {
  const maxDuration = Math.max(...trace.map((entry) => entry.duration_ms), 1);
  return (
    <ol className="relative ml-3 border-l-2 border-niq-border">
      {trace.map((entry, index) => (
        <li key={`${entry.agent_name}-${index}`} className="relative mb-5 pl-6 last:mb-0">
          <span className="absolute -left-[9px] top-1 h-4 w-4 rounded-full border-2 border-white bg-niq-blue" />
          <div className="flex flex-wrap items-baseline gap-x-3">
            <span className="font-bold text-niq-navy">{entry.agent_name}</span>
            <span className="font-mono text-xs text-niq-text-secondary">
              {entry.duration_ms.toLocaleString()} ms
            </span>
          </div>
          <p className="mt-0.5 text-sm text-niq-text-secondary">{entry.output_summary}</p>
          <div className="mt-1.5 h-1.5 w-full max-w-md rounded-full bg-niq-bg">
            <div
              className="h-full rounded-full bg-niq-blue"
              style={{ width: `${Math.max(2, (entry.duration_ms / maxDuration) * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ol>
  );
}
