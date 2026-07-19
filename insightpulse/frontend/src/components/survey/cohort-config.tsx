"use client";

import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { formatNumber } from "@/lib/utils";

export interface CohortSettings {
  size: number;
  ageGroup: string;
  incomeGroup: string;
  region: string;
  archetype: string;
}

interface CohortConfigProps {
  value: CohortSettings;
  onChange: (value: CohortSettings) => void;
  maxSize: number;
}

/** "upper_middle" → "Upper Middle" for filter labels. */
function humanize(value: string): string {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

const ANY = { value: "", label: "Any" };
const AGE_GROUPS = [ANY, ...["18-24", "25-34", "35-44", "45-54", "55-64", "65+"].map((v) => ({ value: v, label: v }))];
const INCOME_GROUPS = [ANY, ...["low", "lower_middle", "middle", "upper_middle", "high"].map((v) => ({ value: v, label: humanize(v) }))];
const REGIONS = [
  ANY,
  ...["northeast", "mid_atlantic", "southeast", "south", "midwest", "mountain", "west", "pacific"].map(
    (v) => ({ value: v, label: humanize(v) }),
  ),
];
const ARCHETYPES = [ANY, ...[
  "price_sensitive",
  "premium_loyalist",
  "category_explorer",
  "promotion_driven",
  "convenience_oriented",
].map((v) => ({ value: v, label: humanize(v) }))];

/** Cohort size + demographic/archetype filter panel (tier-capped size). */
export function CohortConfig({ value, onChange, maxSize }: CohortConfigProps) {
  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="cohort-size">
          Cohort size — {formatNumber(value.size)} respondents
        </Label>
        <Slider
          id="cohort-size"
          min={10}
          max={maxSize}
          step={10}
          value={value.size}
          onChange={(event) => onChange({ ...value, size: Number(event.target.value) })}
        />
        <p className="mt-1 text-xs text-niq-text-secondary">
          Your tier allows up to {formatNumber(maxSize)} respondents per run.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <Label htmlFor="filter-age">Age group</Label>
          <Select
            id="filter-age"
            options={AGE_GROUPS}
            value={value.ageGroup}
            onChange={(event) => onChange({ ...value, ageGroup: event.target.value })}
          />
        </div>
        <div>
          <Label htmlFor="filter-income">Income group</Label>
          <Select
            id="filter-income"
            options={INCOME_GROUPS}
            value={value.incomeGroup}
            onChange={(event) => onChange({ ...value, incomeGroup: event.target.value })}
          />
        </div>
        <div>
          <Label htmlFor="filter-region">Region</Label>
          <Select
            id="filter-region"
            options={REGIONS}
            value={value.region}
            onChange={(event) => onChange({ ...value, region: event.target.value })}
          />
        </div>
        <div>
          <Label htmlFor="filter-archetype">Behavioral archetype</Label>
          <Select
            id="filter-archetype"
            options={ARCHETYPES}
            value={value.archetype}
            onChange={(event) => onChange({ ...value, archetype: event.target.value })}
          />
        </div>
      </div>
    </div>
  );
}
