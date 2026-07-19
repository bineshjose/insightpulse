"""Generate synthetic panelist sample data for demo mode.

Produces three CSV files in ``data/demo/``:

- ``panelists.csv`` — 2,560 households with jointly realistic demographics
  (income correlates with age and education, children with age, etc.) and
  a latent behavioral archetype matching the K=5 clusters in L2, held to
  exact quota counts (25/15/20/25/15%).
- ``purchases.csv`` — 27,520 purchase records (≈10.75 per household) whose
  category mix, price tier, promotion response, and shopping calendar
  (weekday/seasonal weighting across Oct 2024 - Jun 2026) are driven by
  each household's archetype.
- ``survey_responses.csv`` — historical survey answers per household,
  conditioned on archetype and demographics, used as the empirical ground
  truth that digital twins are validated against.

The generator is fully deterministic under DEFAULT_SEED so demo runs are
reproducible (a requirement for the AuditAgent's provenance guarantees).

Usage:
    python -m data.demo.generate_sample_data
    # or: make generate-data
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Generation Constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
NUM_PANELISTS = 2_560
NUM_PURCHASES = 27_520
OUTPUT_DIR = Path(__file__).parent

# Fixed reference date so generated data never drifts between runs.
# Purchase history spans 2024-10-01 .. 2026-06-30 (638 days).
REFERENCE_DATE = date(2026, 6, 30)
HISTORY_DAYS = (REFERENCE_DATE - date(2024, 10, 1)).days + 1

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_WEIGHTS = [0.12, 0.22, 0.20, 0.18, 0.15, 0.13]

# Numeric age sampled within each bracket (panel covers ages 18-85).
AGE_RANGE_BY_GROUP: dict[str, tuple[int, int]] = {
    "18-24": (18, 24),
    "25-34": (25, 34),
    "35-44": (35, 44),
    "45-54": (45, 54),
    "55-64": (55, 64),
    "65+": (65, 85),
}

INCOME_GROUPS = ["low", "lower_middle", "middle", "upper_middle", "high"]

REGIONS = [
    "northeast", "mid_atlantic", "southeast", "south",
    "midwest", "mountain", "west", "pacific",
]
REGION_WEIGHTS = [0.14, 0.10, 0.16, 0.12, 0.15, 0.07, 0.12, 0.14]

HOUSEHOLD_SIZES = ["1", "2", "3-4", "5+"]

EDUCATION_LEVELS = ["high_school", "some_college", "bachelors", "masters", "doctorate"]

# P(education | age group): older cohorts skew toward high school,
# 25-44 cohorts toward college degrees.
EDUCATION_BY_AGE: dict[str, list[float]] = {
    "18-24": [0.35, 0.40, 0.22, 0.03, 0.00],
    "25-34": [0.20, 0.25, 0.37, 0.15, 0.03],
    "35-44": [0.22, 0.26, 0.33, 0.15, 0.04],
    "45-54": [0.28, 0.27, 0.29, 0.13, 0.03],
    "55-64": [0.33, 0.27, 0.26, 0.11, 0.03],
    "65+": [0.40, 0.26, 0.21, 0.10, 0.03],
}

# P(income | education): the strongest single predictor of income bracket.
INCOME_BY_EDUCATION: dict[str, list[float]] = {
    "high_school": [0.30, 0.32, 0.25, 0.10, 0.03],
    "some_college": [0.20, 0.28, 0.30, 0.17, 0.05],
    "bachelors": [0.08, 0.17, 0.30, 0.30, 0.15],
    "masters": [0.04, 0.10, 0.24, 0.35, 0.27],
    "doctorate": [0.03, 0.07, 0.20, 0.33, 0.37],
}

# P(employment | age group).
EMPLOYMENT_STATUSES = ["employed", "unemployed", "student", "retired"]
EMPLOYMENT_BY_AGE: dict[str, list[float]] = {
    "18-24": [0.55, 0.10, 0.35, 0.00],
    "25-34": [0.86, 0.08, 0.06, 0.00],
    "35-44": [0.90, 0.09, 0.01, 0.00],
    "45-54": [0.89, 0.09, 0.00, 0.02],
    "55-64": [0.70, 0.08, 0.00, 0.22],
    "65+": [0.12, 0.02, 0.00, 0.86],
}

# P(has_children | age group) — families concentrate in the 25-54 range.
CHILDREN_PROB_BY_AGE: dict[str, float] = {
    "18-24": 0.15,
    "25-34": 0.45,
    "35-44": 0.62,
    "45-54": 0.38,
    "55-64": 0.10,
    "65+": 0.03,
}

# Behavioral archetypes — the latent classes the K=5 clustering in L2
# is expected to recover. Each defines category preferences, price tier
# weights, promotion affinity, and relative purchase frequency.
PRODUCT_CATEGORIES = [
    "snacks", "beverages", "dairy", "bakery", "frozen_foods",
    "household_care", "personal_care", "produce",
]

PRICE_TIERS = ["budget", "value", "mid", "premium", "luxury"]
# Representative unit price ranges per tier (matches PurchaseRecord.price_bin).
TIER_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "budget": (0.5, 2.0),
    "value": (2.0, 5.0),
    "mid": (5.0, 10.0),
    "premium": (10.0, 20.0),
    "luxury": (20.0, 45.0),
}

ARCHETYPES: dict[str, dict[str, Any]] = {
    "price_sensitive": {
        "category_weights": [0.16, 0.14, 0.15, 0.13, 0.14, 0.12, 0.08, 0.08],
        "tier_weights": [0.38, 0.36, 0.18, 0.06, 0.02],
        "promo_rate": 0.40,
        "frequency_weight": 1.1,
    },
    "premium_loyalist": {
        "category_weights": [0.12, 0.16, 0.14, 0.10, 0.08, 0.10, 0.16, 0.14],
        "tier_weights": [0.03, 0.10, 0.27, 0.38, 0.22],
        "promo_rate": 0.10,
        "frequency_weight": 0.9,
    },
    "category_explorer": {
        "category_weights": [0.28, 0.24, 0.08, 0.12, 0.12, 0.04, 0.08, 0.04],
        "tier_weights": [0.12, 0.28, 0.34, 0.20, 0.06],
        "promo_rate": 0.32,
        "frequency_weight": 1.4,
    },
    "convenience_oriented": {
        "category_weights": [0.08, 0.12, 0.16, 0.08, 0.06, 0.06, 0.16, 0.28],
        "tier_weights": [0.05, 0.18, 0.35, 0.30, 0.12],
        "promo_rate": 0.18,
        "frequency_weight": 1.0,
    },
    "promotion_driven": {
        "category_weights": [0.12, 0.10, 0.16, 0.10, 0.18, 0.18, 0.08, 0.08],
        "tier_weights": [0.22, 0.36, 0.28, 0.11, 0.03],
        "promo_rate": 0.45,
        "frequency_weight": 0.8,
    },
}

# Exact population share per archetype (matches the K=5 cluster profile:
# C0 25% / C1 15% / C2 20% / C3 25% / C4 15%). Enforced as hard quotas so
# the generated panel always lands on these counts exactly.
ARCHETYPE_QUOTAS: dict[str, float] = {
    "price_sensitive": 0.25,
    "premium_loyalist": 0.15,
    "category_explorer": 0.20,
    "convenience_oriented": 0.15,
    "promotion_driven": 0.25,
}

# P(archetype | income group) — premium behavior concentrates in higher
# brackets, value seeking in lower ones. Order matches ARCHETYPES keys.
ARCHETYPE_BY_INCOME: dict[str, list[float]] = {
    "low": [0.357, 0.022, 0.185, 0.069, 0.367],
    "lower_middle": [0.291, 0.044, 0.207, 0.100, 0.358],
    "middle": [0.213, 0.099, 0.236, 0.151, 0.301],
    "upper_middle": [0.123, 0.220, 0.230, 0.221, 0.206],
    "high": [0.061, 0.365, 0.196, 0.254, 0.124],
}

# Recognisable brands per category so demo data reads like real FMCG panel
# records. Order loosely maps to price tier (value first, premium last).
BRANDS_BY_CATEGORY: dict[str, list[str]] = {
    "snacks": ["Lay's", "Pringles", "Doritos", "Cheetos", "Ritz"],
    "beverages": ["Coca-Cola", "Pepsi", "Dr Pepper", "Gatorade", "Tropicana"],
    "dairy": ["Chobani", "Yoplait", "Philadelphia", "Land O'Lakes"],
    "bakery": ["Wonder Bread", "Sara Lee", "Thomas'", "Entenmann's"],
    "frozen_foods": ["Stouffer's", "DiGiorno", "Birds Eye", "Hot Pockets"],
    "household_care": ["Tide", "Clorox", "Dawn", "Febreze"],
    "personal_care": ["Dove", "Colgate", "Pantene", "Nivea"],
    "produce": ["Dole", "Chiquita", "Fresh Express", "Driscoll's"],
}

STORE_TYPES = ["supermarket", "convenience", "online", "warehouse_club", "discount"]
STORE_WEIGHTS = [0.52, 0.14, 0.16, 0.10, 0.08]

# ---------------------------------------------------------------------------
# Sample Survey Definition
# ---------------------------------------------------------------------------

LIKERT_5 = [
    "Not at all important", "Slightly important", "Moderately important",
    "Very important", "Extremely important",
]
AGREE_5 = [
    "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree",
]
CHOICE_DRIVERS = [
    "Price", "Brand reputation", "Quality and ingredients", "Convenience",
    "Promotions and deals",
]
FREQUENCY_OPTIONS = [
    "Daily", "Several times a week", "Weekly", "A few times a month", "Rarely",
]

SURVEY_QUESTIONS: list[dict[str, Any]] = [
    {
        "question_id": "q_organic",
        "text": "How important is organic labeling when purchasing snacks?",
        "question_type": "likert_5",
        "options": LIKERT_5,
    },
    {
        "question_id": "q_nps",
        "text": "How likely are you to recommend your favorite snack brand to a friend?",
        "question_type": "net_promoter",
        "options": [str(i) for i in range(11)],
    },
    {
        "question_id": "q_driver",
        "text": "What matters most when choosing between similar products?",
        "question_type": "single_choice",
        "options": CHOICE_DRIVERS,
    },
    {
        "question_id": "q_sustain",
        "text": "I am willing to pay more for environmentally sustainable products.",
        "question_type": "likert_5",
        "options": AGREE_5,
    },
    {
        "question_id": "q_frequency",
        "text": "How often do you purchase snack foods?",
        "question_type": "single_choice",
        "options": FREQUENCY_OPTIONS,
    },
]

# P(answer | archetype) for each question. These encode the behavioral
# story: health-conscious panelists care about organic labels, value
# seekers pick "Price", impulse buyers snack most often, etc.
ANSWER_WEIGHTS: dict[str, dict[str, list[float]]] = {
    "q_organic": {
        "price_sensitive": [0.34, 0.30, 0.22, 0.10, 0.04],
        "premium_loyalist": [0.06, 0.14, 0.30, 0.32, 0.18],
        "category_explorer": [0.26, 0.30, 0.26, 0.12, 0.06],
        "convenience_oriented": [0.02, 0.06, 0.18, 0.38, 0.36],
        "promotion_driven": [0.22, 0.30, 0.28, 0.14, 0.06],
    },
    "q_nps": {
        # NPS 0-10: loyalists are promoters; value seekers are indifferent.
        "price_sensitive": [0.03, 0.04, 0.06, 0.09, 0.12, 0.16, 0.14, 0.14, 0.11, 0.06, 0.05],
        "premium_loyalist": [0.01, 0.01, 0.02, 0.02, 0.04, 0.06, 0.08, 0.13, 0.21, 0.22, 0.20],
        "category_explorer": [0.02, 0.02, 0.04, 0.06, 0.08, 0.12, 0.14, 0.17, 0.16, 0.11, 0.08],
        "convenience_oriented": [0.02, 0.02, 0.03, 0.05, 0.07, 0.11, 0.13, 0.17, 0.18, 0.13, 0.09],
        "promotion_driven": [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.15, 0.15, 0.13, 0.09, 0.06],
    },
    "q_driver": {
        "price_sensitive": [0.48, 0.06, 0.12, 0.08, 0.26],
        "premium_loyalist": [0.05, 0.34, 0.44, 0.12, 0.05],
        "category_explorer": [0.16, 0.18, 0.16, 0.34, 0.16],
        "convenience_oriented": [0.08, 0.12, 0.58, 0.12, 0.10],
        "promotion_driven": [0.34, 0.08, 0.16, 0.12, 0.30],
    },
    "q_sustain": {
        "price_sensitive": [0.26, 0.32, 0.26, 0.12, 0.04],
        "premium_loyalist": [0.04, 0.10, 0.26, 0.38, 0.22],
        "category_explorer": [0.14, 0.24, 0.34, 0.20, 0.08],
        "convenience_oriented": [0.02, 0.06, 0.16, 0.40, 0.36],
        "promotion_driven": [0.16, 0.28, 0.32, 0.18, 0.06],
    },
    "q_frequency": {
        "price_sensitive": [0.04, 0.18, 0.30, 0.32, 0.16],
        "premium_loyalist": [0.06, 0.20, 0.34, 0.28, 0.12],
        "category_explorer": [0.18, 0.36, 0.26, 0.14, 0.06],
        "convenience_oriented": [0.02, 0.10, 0.24, 0.36, 0.28],
        "promotion_driven": [0.02, 0.12, 0.30, 0.38, 0.18],
    },
}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _archetype_quota_counts(n: int) -> dict[str, int]:
    """Exact archetype counts for a panel of size ``n``.

    Quotas are rounded per archetype; any rounding remainder lands on the
    largest-quota archetype so the counts always sum to ``n``.
    """
    counts = {name: round(n * share) for name, share in ARCHETYPE_QUOTAS.items()}
    counts["price_sensitive"] += n - sum(counts.values())
    return counts


def generate_panelists(rng: np.random.Generator, n: int = NUM_PANELISTS) -> pd.DataFrame:
    """Generate panelist households with jointly realistic demographics.

    Demographics are sampled from conditional distributions rather than
    independently, so the marginals AND the correlations (education→income,
    age→children, income→archetype) look plausible. This matters because
    the BDCL fairness constraints are tested against these joint patterns.

    Archetypes are held to exact quota counts (ARCHETYPE_QUOTAS) while still
    respecting the income correlation: each household samples from
    P(archetype | income) restricted to archetypes with remaining quota.

    Args:
        rng: Seeded random generator.
        n: Number of households to generate.

    Returns:
        DataFrame with one row per household.
    """
    rows: list[dict[str, Any]] = []
    archetype_names = list(ARCHETYPES.keys())
    remaining = _archetype_quota_counts(n)

    for i in range(n):
        age_group = rng.choice(AGE_GROUPS, p=AGE_WEIGHTS)
        age_low, age_high = AGE_RANGE_BY_GROUP[age_group]
        education = rng.choice(EDUCATION_LEVELS, p=EDUCATION_BY_AGE[age_group])
        income_group = rng.choice(INCOME_GROUPS, p=INCOME_BY_EDUCATION[education])
        employment = rng.choice(EMPLOYMENT_STATUSES, p=EMPLOYMENT_BY_AGE[age_group])
        has_children = bool(rng.random() < CHILDREN_PROB_BY_AGE[age_group])

        # Household size follows children status: families with kids are 3+.
        if has_children:
            household_size = rng.choice(["3-4", "5+"], p=[0.72, 0.28])
        else:
            household_size = rng.choice(["1", "2", "3-4"], p=[0.38, 0.48, 0.14])

        # Quota-constrained draw from P(archetype | income): zero out
        # exhausted archetypes and renormalize before sampling.
        weights = np.array([
            p if remaining[name] > 0 else 0.0
            for name, p in zip(archetype_names, ARCHETYPE_BY_INCOME[income_group],
                               strict=True)
        ])
        archetype = str(rng.choice(archetype_names, p=weights / weights.sum()))
        remaining[archetype] -= 1

        join_offset = int(rng.integers(HISTORY_DAYS, 5 * HISTORY_DAYS))
        rows.append({
            "panelist_id": f"HH{i + 1:05d}",
            "age": int(rng.integers(age_low, age_high + 1)),
            "age_group": age_group,
            "income_group": income_group,
            "region": rng.choice(REGIONS, p=REGION_WEIGHTS),
            "household_size": household_size,
            "education_level": education,
            "employment_status": employment,
            "has_children": has_children,
            "behavioral_archetype": archetype,
            "expansion_factor": round(float(rng.lognormal(mean=5.5, sigma=0.4)), 1),
            "panel_join_date": (REFERENCE_DATE - timedelta(days=join_offset)).isoformat(),
        })

    return pd.DataFrame(rows)


def _daily_purchase_weights() -> np.ndarray:
    """Purchase-date sampling weights over the history window.

    Encodes a realistic shopping calendar: weekend peaks, a December
    holiday surge, a January trough, and a mild back-to-school lift —
    so the drift monitor and EDA views see genuine temporal texture
    instead of a uniform smear.
    """
    weekday_factor = {0: 0.95, 1: 0.92, 2: 0.96, 3: 1.00, 4: 1.10, 5: 1.25, 6: 1.12}
    weights = np.empty(HISTORY_DAYS)
    for offset in range(HISTORY_DAYS):
        day = REFERENCE_DATE - timedelta(days=offset)
        factor = weekday_factor[day.weekday()]
        if day.month == 12:
            factor *= 1.30
        elif day.month == 11 and day.day >= 20:
            factor *= 1.20
        elif day.month == 1:
            factor *= 0.88
        elif day.month in (8, 9):
            factor *= 1.05
        weights[offset] = factor
    return weights / weights.sum()


def generate_purchases(
    rng: np.random.Generator,
    panelists: pd.DataFrame,
    n: int = NUM_PURCHASES,
) -> pd.DataFrame:
    """Generate purchase records driven by each household's archetype.

    Purchase volume is allocated across households via a multinomial draw
    weighted by archetype frequency (impulse buyers shop most). Category,
    price tier, and promotion flags follow the archetype profiles, so the
    L2 encoder has real signal to cluster on.

    Args:
        rng: Seeded random generator.
        panelists: Household DataFrame from generate_panelists().
        n: Total number of purchase records.

    Returns:
        DataFrame with one row per transaction.
    """
    freq = panelists["behavioral_archetype"].map(
        {name: spec["frequency_weight"] for name, spec in ARCHETYPES.items()}
    ).to_numpy()
    allocation = rng.multinomial(n, freq / freq.sum())
    date_weights = _daily_purchase_weights()

    rows: list[dict[str, Any]] = []
    for (_, panelist), count in zip(panelists.iterrows(), allocation, strict=True):
        spec = ARCHETYPES[panelist["behavioral_archetype"]]
        day_offsets = rng.choice(HISTORY_DAYS, size=count, p=date_weights)
        for offset in day_offsets:
            category = rng.choice(PRODUCT_CATEGORIES, p=spec["category_weights"])
            tier = rng.choice(PRICE_TIERS, p=spec["tier_weights"])
            low, high = TIER_PRICE_RANGES[tier]
            unit_price = round(float(rng.uniform(low, high)), 2)
            quantity = int(rng.integers(1, 6))
            is_promotion = bool(rng.random() < spec["promo_rate"])
            # Promotions discount the paid price, not the shelf price.
            paid_price = round(unit_price * 0.8, 2) if is_promotion else unit_price

            rows.append({
                "panelist_id": panelist["panelist_id"],
                "transaction_date": (
                    REFERENCE_DATE - timedelta(days=int(offset))
                ).isoformat(),
                "product_category": category,
                "brand": rng.choice(BRANDS_BY_CATEGORY[category]),
                "quantity": quantity,
                "unit_price": paid_price,
                "total_value": round(paid_price * quantity, 2),
                "store_type": rng.choice(STORE_TYPES, p=STORE_WEIGHTS),
                "is_promotion": is_promotion,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["panelist_id", "transaction_date"]).reset_index(drop=True)


def generate_survey_responses(
    rng: np.random.Generator,
    panelists: pd.DataFrame,
) -> pd.DataFrame:
    """Generate historical survey responses conditioned on archetype.

    Every household answers all sample questions. Answer distributions are
    archetype-specific, giving the Validation tab a non-trivial empirical
    ground truth: calibration must recover these conditional patterns, not
    just the population marginals.

    Args:
        rng: Seeded random generator.
        panelists: Household DataFrame from generate_panelists().

    Returns:
        DataFrame with one row per (household, question) answer.
    """
    rows: list[dict[str, Any]] = []
    for _, panelist in panelists.iterrows():
        archetype = panelist["behavioral_archetype"]
        for question in SURVEY_QUESTIONS:
            weights = ANSWER_WEIGHTS[question["question_id"]][archetype]
            answer_index = int(rng.choice(len(question["options"]), p=weights))
            rows.append({
                "response_id": f"R{len(rows) + 1:06d}",
                "panelist_id": panelist["panelist_id"],
                "question_id": question["question_id"],
                "question_text": question["text"],
                "question_type": question["question_type"],
                "answer": question["options"][answer_index],
                "answer_index": answer_index,
                "behavioral_archetype": archetype,
                "response_date": (
                    REFERENCE_DATE - timedelta(days=int(rng.integers(0, HISTORY_DAYS)))
                ).isoformat(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(seed: int = DEFAULT_SEED, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Generate all sample data files.

    Args:
        seed: Random seed for reproducibility.
        output_dir: Directory for the generated CSV files.

    Returns:
        Mapping of dataset name to written file path.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("generation_start", seed=seed, output_dir=str(output_dir))

    panelists = generate_panelists(rng)
    purchases = generate_purchases(rng, panelists)
    responses = generate_survey_responses(rng, panelists)

    paths = {
        "panelists": output_dir / "panelists.csv",
        "purchases": output_dir / "purchases.csv",
        "survey_responses": output_dir / "survey_responses.csv",
    }
    panelists.to_csv(paths["panelists"], index=False)
    purchases.to_csv(paths["purchases"], index=False)
    responses.to_csv(paths["survey_responses"], index=False)

    logger.info(
        "generation_complete",
        panelists=len(panelists),
        purchases=len(purchases),
        survey_responses=len(responses),
        total_spend=round(float(purchases["total_value"].sum()), 2),
        archetype_counts={
            str(k): int(v)
            for k, v in panelists["behavioral_archetype"].value_counts().items()
        },
    )
    return paths


if __name__ == "__main__":
    main()
