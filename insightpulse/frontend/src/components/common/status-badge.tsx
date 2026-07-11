import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

type Status = "healthy" | "warning" | "error";

const CONFIG: Record<Status, { variant: "green" | "amber" | "red"; Icon: typeof CheckCircle2 }> = {
  healthy: { variant: "green", Icon: CheckCircle2 },
  warning: { variant: "amber", Icon: AlertTriangle },
  error: { variant: "red", Icon: XCircle },
};

/** Status badge — icon + label, never color alone. */
export function StatusBadge({ status, label }: { status: Status; label: string }) {
  const { variant, Icon } = CONFIG[status];
  return (
    <Badge variant={variant}>
      <Icon aria-hidden className="h-3 w-3" />
      {label}
    </Badge>
  );
}
