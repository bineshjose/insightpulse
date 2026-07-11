import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  message: string;
  action?: ReactNode;
}

/** Friendly empty state: icon, headline, guidance, optional action. */
export function EmptyState({ icon: Icon, title, message, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-niq-border bg-niq-card px-6 py-14 text-center">
      <Icon aria-hidden className="mb-3 h-10 w-10 text-niq-blue" />
      <h3 className="text-base font-bold text-niq-navy">{title}</h3>
      <p className="mt-1 max-w-md text-sm text-niq-text-secondary">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
