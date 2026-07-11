"use client";

import {
  createContext,
  useContext,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";

/**
 * shadcn-style tabs (underline variant, navy active state).
 * Controlled via internal state; compose Tabs → TabsList → TabsTrigger
 * and one TabsContent per value.
 */

interface TabsContextValue {
  value: string;
  setValue: (value: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabs(): TabsContextValue {
  const context = useContext(TabsContext);
  if (context === null) throw new Error("Tabs components must be used within <Tabs>");
  return context;
}

export function Tabs({
  defaultValue,
  children,
  className,
}: {
  defaultValue: string;
  children: ReactNode;
  className?: string;
}) {
  const [value, setValue] = useState(defaultValue);
  return (
    <TabsContext.Provider value={{ value, setValue }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="tablist"
      className={cn("flex gap-1 border-b-2 border-niq-border", className)}
      {...props}
    />
  );
}

export function TabsTrigger({
  value,
  children,
  className,
}: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  const tabs = useTabs();
  const active = tabs.value === value;
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={() => tabs.setValue(value)}
      className={cn(
        "-mb-0.5 px-4 py-2 text-sm font-semibold transition-colors",
        active
          ? "border-b-[3px] border-niq-navy text-niq-navy"
          : "text-niq-text-secondary hover:text-niq-navy",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function TabsContent({
  value,
  children,
  className,
}: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  const tabs = useTabs();
  if (tabs.value !== value) return null;
  return (
    <div role="tabpanel" className={cn("pt-4", className)}>
      {children}
    </div>
  );
}
