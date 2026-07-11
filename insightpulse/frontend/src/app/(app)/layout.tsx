"use client";

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";
import { Footer } from "@/components/layout/footer";
import { Header } from "@/components/layout/header";
import { Sidebar } from "@/components/layout/sidebar";
import { Skeleton } from "@/components/common/loading-skeleton";
import { useAuth } from "@/lib/auth";

/**
 * Authenticated shell: sidebar + header + content + footer. Redirects to
 * /login when no session exists (after the persisted session restores).
 */
export default function AppLayout({ children }: { children: ReactNode }) {
  const { isAuthenticated, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !isAuthenticated) router.replace("/login");
  }, [loading, isAuthenticated, router]);

  if (loading || !isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-niq-bg">
        <Skeleton className="h-24 w-72" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header />
        <main className="flex-1 px-6 py-6">{children}</main>
        <div className="px-6">
          <Footer />
        </div>
      </div>
    </div>
  );
}
