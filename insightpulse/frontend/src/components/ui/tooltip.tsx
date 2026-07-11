import type { ReactNode } from "react";

interface TooltipProps {
  content: string;
  children: ReactNode;
}

/** Lightweight CSS tooltip (hover/focus), no positioning library. */
export function Tooltip({ content, children }: TooltipProps) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 w-max max-w-56 -translate-x-1/2 rounded-md bg-niq-text px-2.5 py-1.5 text-xs text-white opacity-0 shadow transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {content}
      </span>
    </span>
  );
}
