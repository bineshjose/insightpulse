"""Panelist and household data models.

These models represent the core consumer data that feeds into the digital
twin generation pipeline. Each household has demographic attributes and
purchase history, which together form the conditioning input u_i = [z_i; d_i]
for the generative model.

Corresponds to:
- CIP CPS cloud copy (demographic data)
- BDL CPS purchases (behavioral data)
- RMS product reference (product context)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations for demographic attributes
# ---------------------------------------------------------------------------

class AgeGroup(StrEnum):
    """Age group classification matching NIQ panel structure."""

    AGE_18_24 = "18-24"
    AGE_25_34 = "25-34"
    AGE_35_44 = "35-44"
    AGE_45_54 = "45-54"
    AGE_55_64 = "55-64"
    AGE_65_PLUS = "65+"


class IncomeGroup(StrEnum):
    """Household income brackets."""

    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


class Region(StrEnum):
    """Geographic region classification."""

    NORTHEAST = "northeast"
    MIDWEST = "midwest"
    SOUTH = "south"
    WEST = "west"
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"


class HouseholdSize(StrEnum):
    """Household size categories."""

    SINGLE = "1"
    COUPLE = "2"
    SMALL_FAMILY = "3-4"
    LARGE_FAMILY = "5+"


# ---------------------------------------------------------------------------
# Demographic Profile
# ---------------------------------------------------------------------------

class DemographicProfile(BaseModel):
    """Structured demographic attributes of a panelist household.

    These attributes form the demographic vector D_i that conditions
    the digital twin's survey response generation. Each attribute
    maps to a dimension in the demographic embedding space.
    """

    age_group: AgeGroup
    income_group: IncomeGroup
    region: Region
    household_size: HouseholdSize
    education_level: str = Field(
        description="Highest education level (e.g., 'high_school', 'bachelors', 'masters')"
    )
    has_children: bool = Field(description="Whether the household has children under 18")
    employment_status: str = Field(
        default="employed",
        description="Employment status (employed, unemployed, retired, student)",
    )

    def to_prompt_description(self) -> str:
        """Convert demographics to a natural-language description for LLM prompts.

        This is used in the persona prompt that conditions the digital twin's
        response generation. The description should feel natural and complete
        without revealing the underlying data structure.

        Returns:
            Human-readable demographic description string.
        """
        children_desc = "with children" if self.has_children else "without children"
        return (
            f"A {self.age_group.value} year old {self.employment_status} person "
            f"from a {self.region.value} area, living in a household of "
            f"{self.household_size.value} {children_desc}. "
            f"Education: {self.education_level}. "
            f"Income level: {self.income_group.value}."
        )


# ---------------------------------------------------------------------------
# Purchase Records
# ---------------------------------------------------------------------------

class PurchaseRecord(BaseModel):
    """A single purchase transaction by a panelist household.

    Purchase records are tokenized into behavioral event sequences
    and fed into the transformer encoder to produce behavioral
    embeddings B_i ∈ ℝ^128.
    """

    panelist_id: str = Field(description="Unique household identifier")
    transaction_date: date
    product_category: str = Field(description="Product category (e.g., 'snacks', 'beverages')")
    product_subcategory: str = Field(default="", description="Product subcategory")
    brand: str = Field(default="", description="Product brand name")
    quantity: int = Field(ge=1, description="Number of units purchased")
    unit_price: float = Field(ge=0.0, description="Price per unit in local currency")
    total_value: float = Field(ge=0.0, description="Total transaction value")
    store_type: str = Field(
        default="supermarket",
        description="Store type (supermarket, convenience, online, etc.)",
    )
    is_promotion: bool = Field(default=False, description="Whether purchased on promotion")

    @property
    def price_bin(self) -> str:
        """Categorize unit price into bins for behavioral tokenization.

        Price bins are used as tokens in the purchase sequence fed to
        the transformer encoder. The binning reduces vocabulary size
        while preserving price sensitivity signal.
        """
        if self.unit_price < 2.0:
            return "budget"
        elif self.unit_price < 5.0:
            return "value"
        elif self.unit_price < 10.0:
            return "mid"
        elif self.unit_price < 20.0:
            return "premium"
        else:
            return "luxury"

    def to_behavioral_token(self) -> str:
        """Convert this purchase to a behavioral token for sequence encoding.

        The token format encodes the key dimensions that differentiate
        shopping behaviors: category, price sensitivity, and promotion
        response. These tokens form the input sequence for the
        transformer-based behavioral encoder.

        Returns:
            String token like "snacks|mid|no_promo"
        """
        promo = "promo" if self.is_promotion else "no_promo"
        return f"{self.product_category}|{self.price_bin}|{promo}"


# ---------------------------------------------------------------------------
# Panelist (Individual Member)
# ---------------------------------------------------------------------------

class Panelist(BaseModel):
    """A single panelist (household member) in the consumer panel.

    While surveys are typically administered at the household level,
    individual panelist data enriches the demographic profile and
    enables member-level analysis when needed.
    """

    panelist_id: str = Field(description="Unique panelist identifier")
    household_id: str = Field(description="Parent household identifier")
    role: str = Field(
        default="primary",
        description="Role within household (primary, secondary, child)",
    )
    gender: str = Field(default="unspecified")
    age: int | None = Field(default=None, ge=0, le=120)


# ---------------------------------------------------------------------------
# Household (Primary Unit of Analysis)
# ---------------------------------------------------------------------------

class Household(BaseModel):
    """A panelist household — the primary unit of analysis.

    Each household is a potential synthetic respondent. The household's
    demographic profile and purchase history are combined into the
    conditioning vector u_i that drives digital twin generation.

    Attributes correspond to:
    - demographics: CIP CPS cloud copy data
    - purchases: BDL CPS purchase data
    - members: Household composition data
    - expansion_factor: Statistical weight for population projection
    """

    household_id: str = Field(description="Unique household identifier")
    demographics: DemographicProfile
    members: list[Panelist] = Field(default_factory=list)
    purchases: list[PurchaseRecord] = Field(default_factory=list)
    expansion_factor: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Household expansion factor for population projection. "
            "A factor of 500 means this household represents 500 "
            "households in the target population."
        ),
    )
    panel_join_date: date | None = Field(
        default=None,
        description="Date when household joined the panel",
    )
    last_activity_date: datetime | None = Field(
        default=None,
        description="Most recent purchase or survey activity",
    )

    @property
    def purchase_count(self) -> int:
        """Total number of purchase transactions."""
        return len(self.purchases)

    @property
    def total_spend(self) -> float:
        """Total spending across all purchase records."""
        return sum(p.total_value for p in self.purchases)

    @property
    def unique_categories(self) -> set[str]:
        """Set of unique product categories purchased."""
        return {p.product_category for p in self.purchases}

    @property
    def promotion_rate(self) -> float:
        """Fraction of purchases made on promotion."""
        if not self.purchases:
            return 0.0
        promo_count = sum(1 for p in self.purchases if p.is_promotion)
        return promo_count / len(self.purchases)
