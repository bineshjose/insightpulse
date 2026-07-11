"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { Logo } from "@/components/layout/logo";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/lib/auth";

/**
 * Login page — navy gradient with the subtle pulse-grid pattern, centered
 * card, same credentials as the Streamlit dashboard.
 */
export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (login(email, password)) {
      router.push("/dashboard");
    } else {
      setError("Invalid email or password");
    }
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-niq-navy to-niq-navy-light px-4">
      {/* subtle geometric grid overlay */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage:
            "linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)",
          backgroundSize: "80px 80px",
        }}
      />

      <div className="relative w-full max-w-md">
        <div className="mb-6 flex flex-col items-center">
          <div className="rounded-2xl bg-white/95 px-6 py-4 shadow-xl">
            <Logo tone="dark" className="scale-110" />
          </div>
          <p className="mt-4 text-center text-sm font-medium tracking-[0.18em] text-white/85">
            SYNTHETIC SURVEY INTELLIGENCE PLATFORM
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-2xl bg-white p-7 shadow-2xl"
          aria-label="Sign in"
        >
          <div className="space-y-4">
            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@nielseniq.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                placeholder="••••••••"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </div>
            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 text-niq-text-secondary">
                <Checkbox /> Remember me
              </label>
              <a href="#" className="font-semibold text-niq-blue hover:underline">
                Forgot password?
              </a>
            </div>
            {error && (
              <p role="alert" className="rounded-lg bg-niq-red/10 px-3 py-2 text-sm font-semibold text-niq-red">
                {error}
              </p>
            )}
            <Button type="submit" className="w-full">
              Sign In
            </Button>
          </div>
        </form>

        <p className="mt-6 text-center text-xs leading-relaxed text-white/65">
          IIT Madras &nbsp;×&nbsp; NielsenIQ
          <br />
          M.Tech Industrial AI Project — Binesh Jose (CH24M521)
        </p>
      </div>
    </main>
  );
}
