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
import {
  CATEGORIES,
  CLIENTS,
  PRIORITIES,
  SUPPORTED_MODELS,
  makeSurveyId,
  storeLastRun,
  suggestContractId,
} from "@/lib/demo-data";
import { cn, formatUsd } from "@/lib/utils";
import type { SurveyRunResponse } from "@/lib/types";

const STEPS = ["Setup", "Questions", "Cohort", "Config", "Review & Run"] as const;

/** Rough per-response cost by model family (matches the demo profiles). */
const COST_PER_RESPONSE: Record<string, number> = {
  "claude-sonnet-4-6": 0.0022,
  "gpt-4o": 0.0014,
  "claude-haiku-4-5": 0.0006,
  "ollama/llama3.1": 0,
};

/** Multi-step survey wizard: Setup → Questions → Cohort → Config → Review → Run. */
export default function SurveyPage() {
  const router = useRouter();
  const { user } = useAuth();
  const [step, setStep] = useState(0);
  const [surveyName, setSurveyName] = useState("");
  const [client, setClient] = useState<string>(CLIENTS[0]);
  const [contractId, setContractId] = useState<string>(suggestContractId(CLIENTS[0]));
  const [category, setCategory] = useState<string>(CATEGORIES[0]);
  const [priority, setPriority] = useState<string>("Medium");
  const [dueDate, setDueDate] = useState("");
  const [notes, setNotes] = useState("");
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

  const selectClient = (name: string) => {
    setClient(name);
    setContractId(suggestContractId(name));
  };

  const stepValid = useMemo(() => {
    if (step === 0) return surveyName.trim().length > 0;
    if (step === 1) return validQuestions.length > 0;
    return true;
  }, [step, surveyName, validQuestions.length]);

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
      const enriched: SurveyRunResponse = {
        ...run,
        metadata: {
          survey_id: makeSurveyId(143),
          survey_name: surveyName.trim(),
          client_name: client,
          contract_id: contractId.trim(),
          category,
          region: user?.regions.join(", ") ?? "",
          priority,
          executor_name: user?.name ?? "",
          executor_email: user?.email ?? "",
          created_at: new Date().toISOString(),
          due_date: dueDate || null,
          notes: notes.trim(),
        },
      };
      storeLastRun(enriched);
      setResult(enriched);
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
            {step === 0 && "Name the survey and attach the client engagement."}
            {step === 1 && "Add survey questions — types are auto-detected by the SurveyDesigner."}
            {step === 2 && "Size the cohort and optionally filter by demographics."}
            {step === 3 && "Choose the generating model and reproducibility seed."}
            {step === 4 && "Review the configuration, then run the pipeline."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {step === 0 && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="survey-name">Survey name</Label>
                <input
                  id="survey-name"
                  type="text"
                  value={surveyName}
                  placeholder="e.g. Organic Labeling Importance"
                  onChange={(event) => setSurveyName(event.target.value)}
                  className="h-10 w-full rounded-lg border border-niq-border px-3 text-sm focus:border-niq-blue focus:outline-none"
                />
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <Label htmlFor="client">Client</Label>
                  <Select
                    id="client"
                    options={CLIENTS.map((c) => ({ value: c, label: c }))}
                    value={client}
                    onChange={(event) => selectClient(event.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="contract">Contract ID</Label>
                  <input
                    id="contract"
                    type="text"
                    value={contractId}
                    onChange={(event) => setContractId(event.target.value)}
                    className="h-10 w-full rounded-lg border border-niq-border px-3 font-mono text-sm focus:border-niq-blue focus:outline-none"
                  />
                </div>
                <div>
                  <Label htmlFor="category">Category</Label>
                  <Select
                    id="category"
                    options={CATEGORIES.map((c) => ({ value: c, label: c }))}
                    value={category}
                    onChange={(event) => setCategory(event.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="priority">Priority</Label>
                  <Select
                    id="priority"
                    options={PRIORITIES.map((p) => ({ value: p, label: p }))}
                    value={priority}
                    onChange={(event) => setPriority(event.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="due-date">Due date (optional)</Label>
                  <input
                    id="due-date"
                    type="date"
                    value={dueDate}
                    onChange={(event) => setDueDate(event.target.value)}
                    className="h-10 w-full rounded-lg border border-niq-border px-3 text-sm focus:border-niq-blue focus:outline-none"
                  />
                </div>
              </div>
              <div>
                <Label htmlFor="notes">Notes (optional)</Label>
                <textarea
                  id="notes"
                  value={notes}
                  rows={2}
                  onChange={(event) => setNotes(event.target.value)}
                  className="w-full rounded-lg border border-niq-border px-3 py-2 text-sm focus:border-niq-blue focus:outline-none"
                />
              </div>
              <p className="text-xs text-niq-text-secondary">
                Region: {user?.regions.join(", ")} · Executor: {user?.name} ({user?.email})
              </p>
            </div>
          )}
          {step === 1 && (
            <QuestionForm questions={questions} onChange={setQuestions} maxQuestions={20} />
          )}
          {step === 2 && (
            <CohortConfig
              value={cohort}
              onChange={setCohort}
              maxSize={Math.min(user?.maxCohortSize ?? 1000, 5000)}
            />
          )}
          {step === 3 && (
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
          {step === 4 && !running && !result && (
            <div className="space-y-3 text-sm">
              <ReviewRow label="Survey" value={surveyName.trim() || "—"} />
              <ReviewRow label="Client" value={client} />
              <ReviewRow label="Contract" value={contractId} />
              <ReviewRow label="Category" value={category} />
              <ReviewRow label="Priority" value={priority} />
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
              <ReviewRow label="Estimated cost" value={formatUsd(estimatedCost)} />
              <Button className="mt-2 w-full" onClick={execute}>
                <Rocket className="h-4 w-4" /> Run Survey
              </Button>
            </div>
          )}
          {step === 4 && (running || result) && (
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
                      Is the backend running? Start it with <code>make demo</code>.
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
