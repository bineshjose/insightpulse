import { forwardRef, type LabelHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Form field label. */
export const Label = forwardRef<HTMLLabelElement, LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn("mb-1.5 block text-sm font-semibold text-niq-text", className)}
      {...props}
    />
  ),
);
Label.displayName = "Label";
