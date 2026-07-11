import type { Metadata } from "next";
import type { ReactNode } from "react";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

export const metadata: Metadata = {
  title: "InsightPulse | Synthetic Survey Intelligence",
  description:
    "Agentic AI digital twins for synthetic panelist pulse surveys — IIT Madras × NielsenIQ.",
};

/** Root layout: fonts, theme background, auth context. */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
