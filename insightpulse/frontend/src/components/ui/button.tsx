import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** shadcn-style button themed to NielsenIQ (navy primary, blue hover). */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-niq-blue disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-niq-navy text-white hover:bg-niq-blue",
        secondary:
          "border border-niq-border bg-niq-card text-niq-navy hover:border-niq-blue hover:text-niq-blue",
        ghost: "text-niq-navy hover:bg-niq-bg",
        destructive: "bg-niq-red text-white hover:bg-niq-red/85",
        link: "text-niq-blue underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-5",
        sm: "h-8 px-3 text-xs",
        lg: "h-11 px-7",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  ),
);
Button.displayName = "Button";
