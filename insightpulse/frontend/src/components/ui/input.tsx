import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** shadcn-style text input with NIQ focus ring. */
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-10 w-full rounded-lg border border-niq-border bg-white px-3 text-sm text-niq-text placeholder:text-niq-text-secondary/60 focus:border-niq-blue focus:outline-none focus:ring-1 focus:ring-niq-blue disabled:cursor-not-allowed disabled:bg-niq-bg disabled:text-niq-text-secondary",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";
