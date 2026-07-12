"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  FlaskConical,
  LayoutDashboard,
  Lock,
  ScanSearch,
  Server,
  ShieldCheck,
  Target,
  User as UserIcon,
} from "lucide-react";
import { useState } from "react";
import { Logo } from "@/components/layout/logo";
import { Avatar } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import type { SubscriptionTier, UserRole } from "@/lib/types";

/** Nav entries with the roles allowed to see them (mirrors the Streamlit
 * PAGE_ACCESS map; null = every signed-in role, including the demo
 * account's Survey Analyst role). */
const NAV_ITEMS: readonly {
  href: string;
  label: string;
  icon: typeof LayoutDashboard;
  roles: readonly UserRole[] | null;
}[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, roles: null },
  {
    href: "/data-explorer",
    label: "Data Explorer",
    icon: ScanSearch,
    roles: ["Platform Administrator", "Read-Only Evaluator"],
  },
  { href: "/survey", label: "Survey Runner", icon: Target, roles: null },
  { href: "/results", label: "Results", icon: BarChart3, roles: null },
  {
    href: "/experiments",
    label: "Experiments",
    icon: FlaskConical,
    roles: ["Platform Administrator", "Read-Only Evaluator"],
  },
  {
    href: "/validation",
    label: "Validation",
    icon: ShieldCheck,
    roles: ["Platform Administrator", "Read-Only Evaluator"],
  },
  {
    href: "/audit",
    label: "Audit",
    icon: ClipboardList,
    roles: ["Platform Administrator", "Read-Only Evaluator"],
  },
  {
    href: "/operations",
    label: "Operations",
    icon: Server,
    roles: ["Platform Administrator"],
  },
  { href: "/profile", label: "Profile", icon: UserIcon, roles: null },
];

const TIER_VARIANT: Record<SubscriptionTier, "navy" | "blue" | "green"> = {
  Enterprise: "navy",
  Professional: "blue",
  Academic: "green",
};

/** Collapsible navy sidebar: logo, nav with active states, user widget. */
export function Sidebar() {
  const pathname = usePathname();
  const { user } = useAuth();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "sticky top-0 flex h-screen flex-col bg-gradient-to-b from-niq-navy to-niq-navy-light text-white transition-all",
        collapsed ? "w-[68px]" : "w-60",
      )}
    >
      <div className="flex items-center justify-between p-4">
        <Logo compact={collapsed} />
      </div>

      <nav className="flex-1 space-y-1 px-2.5" aria-label="Primary">
        {NAV_ITEMS.filter(
          (item) => item.roles === null || (user && item.roles.includes(user.role)),
        ).map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-semibold transition-colors",
                active
                  ? "bg-niq-blue/25 text-white"
                  : "text-white/75 hover:bg-white/10 hover:text-white",
              )}
            >
              <Icon className="h-4.5 w-4.5 h-[18px] w-[18px] shrink-0" />
              {!collapsed && label}
            </Link>
          );
        })}
      </nav>

      {user && (
        <div
          className={cn(
            "space-y-1.5 border-t border-white/20 px-3 py-2.5 text-xs font-semibold text-white/80",
            collapsed && "flex flex-col items-center",
          )}
        >
          <div className="flex items-center gap-2" title="PromptGuard: Active">
            <span aria-hidden className="h-2 w-2 shrink-0 rounded-full bg-niq-green" />
            {!collapsed && (
              <span className="flex items-center gap-1">
                <Lock className="h-3 w-3" /> PromptGuard: Active
              </span>
            )}
          </div>
          {user.role === "Platform Administrator" && (
            <Link
              href="/operations"
              title="System: Healthy"
              className="flex items-center gap-2 hover:text-white"
            >
              <span aria-hidden className="h-2 w-2 shrink-0 rounded-full bg-niq-green" />
              {!collapsed && <span>System: Healthy</span>}
            </Link>
          )}
        </div>
      )}

      {user && (
        <div className="border-t border-white/20 p-3">
          <div className="flex items-center gap-2.5">
            <Avatar initials={user.initials} tier={user.tier} className="border-2 border-white/60" />
            {!collapsed && (
              <div className="min-w-0 leading-tight">
                <div className="truncate text-sm font-bold">{user.name}</div>
                <div className="truncate text-xs text-white/70">{user.role}</div>
                <Badge variant={TIER_VARIANT[user.tier]} className="mt-1 border border-white/40">
                  {user.tier}
                </Badge>
              </div>
            )}
          </div>
        </div>
      )}

      <button
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        onClick={() => setCollapsed((v) => !v)}
        className="flex items-center justify-center border-t border-white/20 py-2.5 text-white/70 hover:text-white"
      >
        {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
      </button>
    </aside>
  );
}
