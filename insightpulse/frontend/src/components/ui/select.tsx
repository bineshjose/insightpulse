import { forwardRef, type SelectHTMLAttributes } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  options: readonly { value: string; label: string }[];
}

/** Styled native select — accessible, keyboard-friendly, zero JS overhead. */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, options, ...props }, ref) => (
    <div className="relative">
      <select
        ref={ref}
        className={cn(
          "h-10 w-full appearance-none rounded-lg border border-niq-border bg-white px-3 pr-9 text-sm text-niq-text focus:border-niq-blue focus:outline-none focus:ring-1 focus:ring-niq-blue disabled:bg-niq-bg",
          className,
        )}
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <ChevronDown
        aria-hidden
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-niq-text-secondary"
      />
    </div>
  ),
);
Select.displayName = "Select";
