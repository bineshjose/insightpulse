"""Shared domain constants (single source — never hardcode these inline).

Values that multiple layers must agree on: dataset identifiers, the
behavioral archetype labels discovered by L2 clustering (K=5), and the
demographic scale orders used for encoding and display.
"""

from __future__ import annotations

# Dataset identifiers (also the SQL table names in demo/production stores).
DATASET_PANELISTS = "panelists"
DATASET_PURCHASES = "purchases"
DATASET_SURVEY_RESPONSES = "survey_responses"

# K=5 behavioral archetypes from the L2 clustering (cluster id -> label).
ARCHETYPE_LABELS: dict[int, str] = {
    0: "Price Sensitive",
    1: "Premium Loyalist",
    2: "Category Explorer",
    3: "Promotion Driven",
    4: "Convenience Oriented",
}

# Ordinal demographic scales (order carries meaning for encoding).
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
INCOME_GROUPS = ["low", "lower_middle", "middle", "upper_middle", "high"]
HOUSEHOLD_SIZES = ["1", "2", "3-4", "5+"]
EDUCATION_LEVELS = ["high_school", "some_college", "bachelors", "masters", "doctorate"]

# Price tiers matching PurchaseRecord.price_bin.
PRICE_TIERS = ["budget", "value", "mid", "premium", "luxury"]
