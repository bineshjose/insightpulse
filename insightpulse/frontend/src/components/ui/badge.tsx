import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Pill badge; tier/status variants map to the NIQ palette. */
const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-bold",
  {
    variants: {
      variant: {
        navy: "bg-niq-navy text-white",
        blue: "bg-niq-blue text-white",
        green: "bg-niq-green text-white",
        amber: "bg-niq-amber text-white",
        red: "bg-niq-red text-white",
        outline: "border border-niq-border text-niq-text-secondary",
      },
    },
    defaultVariants: { variant: "navy" },
  },
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
