"use client";

import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { hasPermission, useAuth } from "@/lib/auth";

interface ExportButtonProps {
  /** File name without extension. */
  filename: string;
  /** Rows for CSV export (also serialized for JSON). */
  data: object[] | object;
  format: "csv" | "json";
}

function toCsv(rows: object[]): string {
  if (rows.length === 0) return "";
  const first = rows[0] as Record<string, unknown>;
  const headers = Object.keys(first);
  const lines = rows.map((row) =>
    headers
      .map((header) => {
        const value = (row as Record<string, unknown>)[header];
        const text = value === null || value === undefined ? "" : String(value);
        return text.includes(",") || text.includes('"')
          ? `"${text.replaceAll('"', '""')}"`
          : text;
      })
      .join(","),
  );
  return [headers.join(","), ...lines].join("\n");
}

/**
 * Download button gated on the 'export' permission — disabled (with an
 * explanatory tooltip) for tiers without export access.
 */
export function ExportButton({ filename, data, format }: ExportButtonProps) {
  const { user } = useAuth();
  const allowed = hasPermission(user, "export");

  const handleDownload = () => {
    const content =
      format === "csv" && Array.isArray(data)
        ? toCsv(data)
        : JSON.stringify(data, null, 2);
    const blob = new Blob([content], {
      type: format === "csv" ? "text/csv" : "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${filename}.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const button = (
    <Button variant="secondary" size="sm" disabled={!allowed} onClick={handleDownload}>
      <Download className="h-4 w-4" />
      {format.toUpperCase()}
    </Button>
  );

  if (allowed) return button;
  return (
    <Tooltip content={`Export requires the 'export' capability — not included in the ${user?.tier ?? ""} tier.`}>
      {button}
    </Tooltip>
  );
}
