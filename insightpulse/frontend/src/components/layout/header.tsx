"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { LogOut, Settings, User as UserIcon } from "lucide-react";
import { useEffect, useState } from "react";
import {
  DropdownMenu,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Avatar } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { getHealth } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { formatNumber } from "@/lib/utils";

const CRUMB_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  survey: "Survey Runner",
  results: "Results",
  experiments: "Experiments",
  validation: "Validation",
  audit: "Audit",
  profile: "Profile",
};

/** Top bar: breadcrumbs, environment badge, credit pill, user menu. */
export function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [env, setEnv] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((health) => setEnv(health.env))
      .catch(() => setEnv(null));
  }, []);

  const segment = pathname.split("/")[1] ?? "";
  const crumb = CRUMB_LABELS[segment] ?? "Dashboard";

  return (
    <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-niq-border bg-niq-card px-6">
      <nav aria-label="Breadcrumb" className="text-sm text-niq-text-secondary">
        <Link href="/dashboard" className="hover:text-niq-blue">
          Home
        </Link>
        <span className="mx-1.5">›</span>
        <span className="font-semibold text-niq-navy">{crumb}</span>
      </nav>

      <div className="flex items-center gap-3">
        {env !== null ? (
          <Badge variant={env === "production" ? "red" : "green"}>
            {env === "production" ? "Live" : "Demo"}
          </Badge>
        ) : (
          <Badge variant="outline">API offline</Badge>
        )}
        {user && (
          <span className="rounded-full border border-niq-border bg-niq-bg px-3 py-1 text-xs font-semibold text-niq-navy">
            {formatNumber(user.creditsBalance)} credits
          </span>
        )}
        {user && (
          <DropdownMenu
            trigger={<Avatar initials={user.initials} tier={user.tier} className="h-8 w-8 text-xs" />}
          >
            <div className="px-3 py-2">
              <div className="text-sm font-bold text-niq-text">{user.name}</div>
              <div className="text-xs text-niq-text-secondary">{user.email}</div>
            </div>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => router.push("/profile")}>
              <UserIcon className="h-4 w-4" /> Profile
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => router.push("/profile")}>
              <Settings className="h-4 w-4" /> Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              destructive
              onSelect={() => {
                logout();
                router.push("/login");
              }}
            >
              <LogOut className="h-4 w-4" /> Logout
            </DropdownMenuItem>
          </DropdownMenu>
        )}
      </div>
    </header>
  );
}
