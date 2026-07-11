import { cn } from "@/lib/utils";
import type { SubscriptionTier } from "@/lib/types";

const TIER_BG: Record<SubscriptionTier, string> = {
  Enterprise: "bg-niq-navy",
  Professional: "bg-niq-blue",
  Academic: "bg-niq-green",
};

interface AvatarProps {
  initials: string;
  tier: SubscriptionTier;
  className?: string;
}

/** Initials avatar, colored by subscription tier (matches Streamlit). */
export function Avatar({ initials, tier, className }: AvatarProps) {
  return (
    <div
      aria-hidden
      className={cn(
        "flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-sm font-bold text-white",
        TIER_BG[tier],
        className,
      )}
    >
      {initials}
    </div>
  );
}
