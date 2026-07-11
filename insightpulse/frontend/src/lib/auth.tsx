"use client";

/**
 * Client-side auth context with the same three demo accounts as the
 * Streamlit dashboard (components/auth.py). Deliberately NOT a security
 * boundary — the production tool authenticates at the ingress; this
 * context provides the session + role model the UI is built around.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Permission, User } from "@/lib/types";

interface Account extends User {
  password: string;
}

/** Demo accounts — identical to the Streamlit fixture. */
const ACCOUNTS: Account[] = [
  {
    email: "binesh.jose@nielseniq.com",
    password: "Ch24m521",
    name: "Binesh Jose",
    initials: "BJ",
    role: "Platform Administrator",
    title: "ML Engineer — Consumer Intelligence",
    department: "Data Science & Analytics",
    regions: ["APAC", "EMEA", "Americas"],
    permissions: ["view", "create", "run", "analyze", "export", "calibrate"],
    tier: "Enterprise",
    creditsTotal: 3000,
    creditsBalance: 2500,
    apiCallsRemaining: 10000,
    apiCallsQuota: 10000,
    maxCohortSize: 5000,
    created: "2025-08-14",
  },
  {
    email: "evaluator@iitm.ac.in",
    password: "eval2024",
    name: "External Evaluator",
    initials: "EV",
    role: "Read-Only Analyst",
    title: "Faculty Reviewer",
    department: "Academic Review Board",
    regions: ["APAC"],
    permissions: ["view", "analyze"],
    tier: "Academic",
    creditsTotal: 600,
    creditsBalance: 500,
    apiCallsRemaining: 1000,
    apiCallsQuota: 1000,
    maxCohortSize: 200,
    created: "2025-11-02",
  },
  {
    email: "demo@insightpulse.ai",
    password: "demo123",
    name: "Demo User",
    initials: "DU",
    role: "Survey Analyst",
    title: "Consumer Research Analyst",
    department: "Market Research",
    regions: ["Americas"],
    permissions: ["view", "create", "run", "analyze"],
    tier: "Professional",
    creditsTotal: 1200,
    creditsBalance: 1000,
    apiCallsRemaining: 5000,
    apiCallsQuota: 5000,
    maxCohortSize: 1000,
    created: "2025-09-30",
  },
];

const STORAGE_KEY = "insightpulse.session";

interface AuthContextValue {
  user: User | null;
  /** True while the persisted session is being restored on first mount. */
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** Wraps the app; restores the session from localStorage on mount. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const email = JSON.parse(stored) as string;
        const account = ACCOUNTS.find((a) => a.email === email);
        if (account) {
          const { password: _password, ...profile } = account;
          setUser(profile);
        }
      }
    } catch {
      // corrupt storage: start signed out
    }
    setLoading(false);
  }, []);

  const login = useCallback((email: string, password: string): boolean => {
    const account = ACCOUNTS.find(
      (a) => a.email === email.trim().toLowerCase() && a.password === password,
    );
    if (!account) return false;
    const { password: _password, ...profile } = account;
    setUser(profile);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile.email));
    return true;
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    window.localStorage.removeItem(STORAGE_KEY);
  }, []);

  const value = useMemo(
    () => ({ user, loading, isAuthenticated: user !== null, login, logout }),
    [user, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** Access the auth session; must be used under AuthProvider. */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used within <AuthProvider>");
  }
  return context;
}

/** Role-based access check used by pages and action buttons. */
export function hasPermission(user: User | null, permission: Permission): boolean {
  return user?.permissions.includes(permission) ?? false;
}
