import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Styled native checkbox (navy accent). */
export const Checkbox = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      type="checkbox"
      className={cn("h-4 w-4 rounded border-niq-border accent-niq-navy", className)}
      {...props}
    />
  ),
);
Checkbox.displayName = "Checkbox";
