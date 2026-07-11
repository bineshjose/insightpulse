import type { Config } from "tailwindcss";

/**
 * NielsenIQ theme tokens. Components reference these (`bg-niq-navy`,
 * `text-niq-blue`, ...) — never raw hex — so the palette changes in one place.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "niq-navy": "#003865",
        "niq-navy-light": "#004A7C",
        "niq-blue": "#00A4E4",
        "niq-green": "#6CC24A",
        "niq-amber": "#F2A900",
        "niq-red": "#E03C31",
        "niq-bg": "#F7F8FA",
        "niq-card": "#FFFFFF",
        "niq-text": "#1A1A2E",
        "niq-text-secondary": "#6B7280",
        "niq-border": "#E5E7EB",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 1px 3px rgba(16, 24, 40, 0.06)",
      },
    },
  },
  plugins: [],
};

export default config;
