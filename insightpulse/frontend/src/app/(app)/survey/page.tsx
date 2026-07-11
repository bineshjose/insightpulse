"use client";

import { useRouter } from "next/navigation";
import { AlertTriangle, ChevronLeft, ChevronRight, Lock, Rocket } from "lucide-react";
import { useMemo, useState } from "react";
import { RunProgress } from "@/components/survey/run-progress";
import { CohortConfig, type CohortSettings } from "@/components/survey/cohort-config";
import { QuestionForm } from "@/components/survey/question-form";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { ApiError, runSurvey } from "@/lib/api";
import { hasPermission, useAuth } from "@/lib/auth";
import { SUPPORTED_MODELS, storeLastRun } from "@/lib/demo-data";
import { cn, formatUsd } from "@/lib/utils";
import type { SurveyRunResponse } from "@/lib/types";

const STEPS = ["Questions", "Cohort", "Config", "Review & Run"] as const;

/** Rough per-response cost by model family (matches simulation profiles). */
const COST_PER_RESPONSE: Record<string, number> = {
  "claude-sonnet-4-6": 0.0022,
  "gpt-4o": 0.0014,
  "claude-haiku-4-5": 0.0006,
  "ollama/llama3.1": 0,
};

/** Multi-step survey wizard: Questions → Cohort → Config → Review → Run. */
export default function SurveyPage() {
  const router = useRouter();
  const { user } = useAuth();
  const [step, setStep] = useState(0);
  const [questions, setQuestions] = useState<string[]>([
    "How important is organic labeling when purchasing snacks?",
  ]);
  const [cohort, setCohort] = useState<CohortSettings>({
    size: 100,
    ageGroup: "",
    incomeGroup: "",
    region: "",
  });
  const [model, setModel] = useState<string>(SUPPORTED_MODELS[0]);
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SurveyRunResponse | null>(null);

  const canRun = hasPermission(user, "run");
  const validQuestions = questions.map((q) => q.trim()).filter(Boolean);
  const estimatedCost = cohort.size * validQuestions.length * (COST_PER_RESPONSE[model] ?? 0.002);

  const stepValid = useMemo(() => {
    if (step === 0) return validQuestions.length > 0;
    return true;
  }, [step, validQuestions.length]);

  if (!canRun) {
    return (
      <Card>
        <CardContent className="flex items-center gap-3 py-8">
          <Lock className="h-6 w-6 text-niq-amber" />
          <div>
            <p className="font-bold text-niq-navy">Access Restricted</p>
            <p className="text-sm text-niq-text-secondary">
              Your role ({user?.role}) does not include the &apos;run&apos; capability —
              contact your administrator.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const execute = async () => {
    setRunning(true);
    setError(null);
    const filters: Record<string, string> = {};
    if (cohort.ageGroup) filters.age_group = cohort.ageGroup;
    if (cohort.incomeGroup) filters.income_group = cohort.incomeGroup;
    if (cohort.region) filters.region = cohort.region;
    try {
      const run = await runSurvey({
        questions: validQuestions,
        cohort_size: cohort.size,
        context: "US consumer goods market",
        models: [model],
        cohort_filters: filters,
        seed,
      });
      storeLastRun(run);
      setResult(run);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.detail
          : "Unexpected error — check the backend logs.",
      );
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-niq-navy">Survey Runner</h1>
        <p className="text-sm text-niq-text-secondary">
          Configure a pulse survey and execute it against digital-twin respondents.
        </p>
      </div>

      {/* Step indicator */}
      <ol className="flex flex-wrap gap-2" aria-label="Wizard steps">
        {STEPS.map((label, index) => (
          <li
            key={label}
            className={cn(
              "flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-bold",
              index === step
                ? "bg-niq-navy text-white"
                : index < step
                  ? "bg-niq-green/15 text-niq-green"
                  : "bg-niq-border/60 text-niq-text-secondary",
            )}
          >
            {index + 1}. {label}
          </li>
        ))}
      </ol>

      <Card>
        <CardHeader>
          <CardTitle>{STEPS[step]}</CardTitle>
          <CardDescription>
            {step === 0 && "Add survey questions — types are auto-detected by the SurveyDesigner."}
            {step === 1 && "Size the cohort and optionally filter by demographics."}
            {step === 2 && "Choose the generating model and reproducibility seed."}
            {step === 3 && "Review the configuration, then run the 8-agent pipeline."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {step === 0 && (
            <QuestionForm questions={questions} onChange={setQuestions} maxQuestions={20} />
          )}
          {step === 1 && (
            <CohortConfig
              value={cohort}
              onChange={setCohort}
              maxSize={Math.min(user?.maxCohortSize ?? 1000, 5000)}
            />
          )}
          {step === 2 && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="model">LLM model</Label>
                <Select
                  id="model"
                  options={SUPPORTED_MODELS.map((m) => ({ value: m, label: m }))}
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                />
                <p className="mt-1 text-xs text-niq-text-secondary">
                  Estimated cost: <b>{formatUsd(estimatedCost)}</b> for{" "}
                  {cohort.size * validQuestions.length} responses
                  {model === "ollama/llama3.1" && " (local model — free)"}
                </p>
              </div>
              <div>
                <Label htmlFor="seed">Random seed</Label>
                <input
                  id="seed"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(event) => setSeed(Math.max(0, Number(event.target.value)))}
                  className="h-10 w-40 rounded-lg border border-niq-border px-3 text-sm focus:border-niq-blue focus:outline-none"
                />
              </div>
            </div>
          )}
          {step === 3 && !running && !result && (
            <div className="space-y-3 text-sm">
              <ReviewRow label="Questions" value={`${validQuestions.length}`} />
              <ReviewRow label="Cohort size" value={`${cohort.size} respondents`} />
              <ReviewRow
                label="Filters"
                value={
                  [cohort.ageGroup, cohort.incomeGroup, cohort.region].filter(Boolean).join(", ") ||
                  "none"
                }
              />
              <ReviewRow label="Model" value={model} />
              <ReviewRow label="Seed" value={String(seed)} />
              <ReviewRow label="Estimated cost" value={formatUsd(estimatedCost)} />
              <Button className="mt-2 w-full" onClick={execute}>
                <Rocket className="h-4 w-4" /> Run Survey
              </Button>
            </div>
          )}
          {step === 3 && (running || result) && (
            <div className="space-y-4">
              <RunProgress done={result !== null} />
              {result && (
                <div className="rounded-lg bg-niq-green/10 p-4 text-sm">
                  <p className="font-bold text-niq-green">
                    Survey completed — {result.total_responses} responses
                  </p>
                  <p className="mt-1 text-niq-text-secondary">
                    Hallucination rate {(result.hallucination_rate * 100).toFixed(1)}% · cost{" "}
                    {formatUsd(result.total_cost_usd)} · provenance{" "}
                    <span className="font-mono">{result.provenance_hash}</span>
                  </p>
                  <Button className="mt-3" onClick={() => router.push("/results")}>
                    View full results
                  </Button>
                </div>
              )}
              {error && (
                <div className="flex items-start gap-2 rounded-lg bg-niq-red/10 p-4 text-sm">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-niq-red" />
                  <div>
                    <p className="font-bold text-niq-red">Run failed</p>
                    <p className="text-niq-text-secondary">{error}</p>
                    <p className="mt-1 text-xs text-niq-text-secondary">
                      Is the backend running? Start it with <code>make demo</code> — demo mode
                      needs no API keys.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button
          variant="secondary"
          disabled={step === 0 || running}
          onClick={() => {
            setResult(null);
            setError(null);
            setStep((s) => Math.max(0, s - 1));
          }}
        >
          <ChevronLeft className="h-4 w-4" /> Back
        </Button>
        {step < STEPS.length - 1 && (
          <Button disabled={!stepValid} onClick={() => setStep((s) => s + 1)}>
            Next <ChevronRight className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}

function ReviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-niq-border pb-2 last:border-0">
      <span className="text-niq-text-secondary">{label}</span>
      <span className="font-semibold text-niq-text">{value}</span>
    </div>
  );
}
