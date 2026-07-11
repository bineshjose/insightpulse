"use client";

import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

/** Minimal modal dialog (backdrop click / Escape / X to close). */
export function Dialog({ open, onClose, title, children }: DialogProps) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-niq-text/50 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal
        aria-label={title}
        className="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-bold text-niq-navy">{title}</h3>
          <button aria-label="Close" onClick={onClose}>
            <X className="h-5 w-5 text-niq-text-secondary hover:text-niq-text" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
