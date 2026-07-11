/**
 * Data Explorer fixtures — the exact aggregates the Streamlit Data
 * Explorer computes from the active panel (data/demo), so both
 * dashboards show identical values.
 */

export const PANEL_SUMMARY = {
  totalHouseholds: 500,
  totalMembers: 1311,
  avgHouseholdSize: 2.62,
  regionsCovered: 4,
} as const;

export const AGE_DISTRIBUTION = [
  { group: "18-24", households: 67, share: 0.134 },
  { group: "25-34", households: 104, share: 0.208 },
  { group: "35-44", households: 112, share: 0.224 },
  { group: "45-54", households: 90, share: 0.18 },
  { group: "55-64", households: 75, share: 0.15 },
  { group: "65+", households: 52, share: 0.104 },
] as const;

export const INCOME_DISTRIBUTION = [
  { group: "Low", households: 87, share: 0.174 },
  { group: "Lower Middle", households: 115, share: 0.23 },
  { group: "Middle", households: 125, share: 0.25 },
  { group: "Upper Middle", households: 115, share: 0.23 },
  { group: "High", households: 58, share: 0.116 },
] as const;

export const REGION_DISTRIBUTION = [
  { group: "South", households: 200, share: 0.4 },
  { group: "West", households: 114, share: 0.228 },
  { group: "Midwest", households: 107, share: 0.214 },
  { group: "Northeast", households: 79, share: 0.158 },
] as const;

export const HOUSEHOLD_SIZE_DISTRIBUTION = [
  { group: "1", households: 121, share: 0.242 },
  { group: "2", households: 160, share: 0.32 },
  { group: "3-4", households: 167, share: 0.334 },
  { group: "5+", households: 52, share: 0.104 },
] as const;

export const PURCHASE_SUMMARY = {
  totalTransactions: 10000,
  avgBasketValue: 24.19,
  avgUnitPrice: 8.14,
  promotionRate: 0.3121,
} as const;

export const CATEGORY_PENETRATION = [
  { category: "Dairy", share: 0.914 },
  { category: "Beverages", share: 0.898 },
  { category: "Snacks", share: 0.896 },
  { category: "Frozen Foods", share: 0.888 },
  { category: "Bakery", share: 0.88 },
  { category: "Personal Care", share: 0.844 },
  { category: "Produce", share: 0.82 },
  { category: "Household Care", share: 0.816 },
] as const;

export const PRICE_TIER_DISTRIBUTION = [
  { tier: "Budget", transactions: 1962 },
  { tier: "Value", transactions: 2773 },
  { tier: "Mid", transactions: 2738 },
  { tier: "Premium", transactions: 1798 },
  { tier: "Luxury", transactions: 729 },
] as const;

export const PROMOTION_BY_ARCHETYPE = [
  { archetype: "Promotion Driven", share: 0.4536 },
  { archetype: "Price Sensitive", share: 0.3953 },
  { archetype: "Category Explorer", share: 0.3064 },
  { archetype: "Convenience Oriented", share: 0.1855 },
  { archetype: "Premium Loyalist", share: 0.0994 },
] as const;

export const MONTHLY_VOLUME = [
  { month: "2025-07", transactions: 828 },
  { month: "2025-08", transactions: 830 },
  { month: "2025-09", transactions: 802 },
  { month: "2025-10", transactions: 872 },
  { month: "2025-11", transactions: 842 },
  { month: "2025-12", transactions: 842 },
  { month: "2026-01", transactions: 808 },
  { month: "2026-02", transactions: 780 },
  { month: "2026-03", transactions: 878 },
  { month: "2026-04", transactions: 866 },
  { month: "2026-05", transactions: 838 },
  { month: "2026-06", transactions: 814 },
] as const;

export const CLUSTER_PROFILES = [
  { name: "Price Sensitive", share: 0.25, households: 125, silhouette: 0.47 },
  { name: "Promotion Driven", share: 0.25, households: 125, silhouette: 0.43 },
  { name: "Category Explorer", share: 0.202, households: 101, silhouette: 0.36 },
  { name: "Convenience Oriented", share: 0.152, households: 76, silhouette: 0.39 },
  { name: "Premium Loyalist", share: 0.146, households: 73, silhouette: 0.44 },
] as const;

export const SILHOUETTE_OVERALL = 0.42;

/** Radar dimensions and per-archetype behavioral signatures (0-1). */
export const ARCHETYPE_RADAR = {
  dimensions: [
    "Price Sensitivity",
    "Brand Loyalty",
    "Category Breadth",
    "Promotion Response",
    "Convenience Preference",
  ],
  profiles: {
    "Price Sensitive": [0.92, 0.25, 0.45, 0.7, 0.35],
    "Premium Loyalist": [0.18, 0.95, 0.35, 0.22, 0.55],
    "Category Explorer": [0.45, 0.3, 0.95, 0.5, 0.48],
    "Promotion Driven": [0.75, 0.35, 0.55, 0.95, 0.4],
    "Convenience Oriented": [0.35, 0.55, 0.4, 0.3, 0.94],
  },
} as const;

export const QUALITY_GATES = [
  { gate: "Null rate", value: "0.0%", target: "< 2%", pass: true },
  { gate: "Primary keys", value: "Unique", target: "No duplicates", pass: true },
  { gate: "Date range", value: "Jul 2025 – Jun 2026", target: "Valid", pass: true },
  { gate: "Distributions", value: "Stable", target: "Top share < 80%", pass: true },
] as const;
