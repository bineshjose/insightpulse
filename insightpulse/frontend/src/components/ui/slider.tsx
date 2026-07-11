import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Styled native range slider (navy accent). */
export const Slider = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      type="range"
      className={cn("h-2 w-full cursor-pointer accent-niq-navy", className)}
      {...props}
    />
  ),
);
Slider.displayName = "Slider";
