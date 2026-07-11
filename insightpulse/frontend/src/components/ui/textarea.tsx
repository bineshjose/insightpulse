import { forwardRef, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Styled textarea with NIQ focus ring. */
export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "w-full rounded-lg border border-niq-border bg-white px-3 py-2 text-sm text-niq-text placeholder:text-niq-text-secondary/60 focus:border-niq-blue focus:outline-none focus:ring-1 focus:ring-niq-blue",
      className,
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";
