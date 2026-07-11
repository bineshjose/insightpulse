import { cn } from "@/lib/utils";

interface LogoProps {
  /** Icon-only (sidebar collapsed) or full wordmark. */
  compact?: boolean;
  className?: string;
  /** Wordmark color — white for the navy sidebar, navy for light surfaces. */
  tone?: "light" | "dark";
}

/** InsightPulse logo: pulse-wave + bars icon with optional wordmark. */
export function Logo({ compact = false, className, tone = "light" }: LogoProps) {
  const wordmark = tone === "light" ? "text-white" : "text-niq-navy";
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <svg
        viewBox="0 0 108 108"
        className="h-9 w-9 shrink-0"
        role="img"
        aria-label="InsightPulse"
      >
        <rect x="4" y="4" width="100" height="100" rx="22" fill="#003865" stroke="#FFFFFF" strokeOpacity="0.25" strokeWidth="2" />
        <rect x="24" y="56" width="12" height="28" rx="3" fill="#FFFFFF" opacity="0.35" />
        <rect x="42" y="44" width="12" height="40" rx="3" fill="#FFFFFF" opacity="0.55" />
        <rect x="60" y="34" width="12" height="50" rx="3" fill="#FFFFFF" opacity="0.75" />
        <rect x="78" y="24" width="12" height="60" rx="3" fill="#FFFFFF" opacity="0.9" />
        <path
          d="M14 52 L34 52 L44 28 L58 70 L68 42 L76 52 L94 52"
          fill="none"
          stroke="#00A4E4"
          strokeWidth="7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      {!compact && (
        <span className={cn("text-lg font-bold leading-none", wordmark)}>
          Insight<span className="text-niq-blue">Pulse</span>
        </span>
      )}
    </span>
  );
}
