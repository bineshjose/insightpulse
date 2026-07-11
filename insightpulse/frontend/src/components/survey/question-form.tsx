"use client";

import { Plus, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface QuestionFormProps {
  questions: string[];
  onChange: (questions: string[]) => void;
  maxQuestions?: number;
}

/** Question-type auto-detection matching the SurveyDesigner heuristics. */
export function detectQuestionType(text: string): string {
  const lowered = text.toLowerCase();
  if (lowered.includes("how likely") || lowered.includes("recommend")) return "net_promoter";
  if (["agree", "disagree", "important", "satisfied"].some((k) => lowered.includes(k)))
    return "likert_5";
  if (["rank", "order", "prioritize"].some((k) => lowered.includes(k))) return "ranking";
  return "single_choice";
}

/** Add/edit survey questions with live type auto-detection badges. */
export function QuestionForm({ questions, onChange, maxQuestions = 20 }: QuestionFormProps) {
  const update = (index: number, text: string) => {
    const next = [...questions];
    next[index] = text;
    onChange(next);
  };

  return (
    <div className="space-y-3">
      {questions.map((question, index) => (
        <div key={index} className="flex items-start gap-2">
          <div className="flex-1">
            <Textarea
              rows={2}
              value={question}
              placeholder={`Question ${index + 1} — e.g. "How important is organic labeling when purchasing snacks?"`}
              onChange={(event) => update(index, event.target.value)}
            />
            {question.trim() && (
              <Badge variant="outline" className="mt-1">
                auto-detected: {detectQuestionType(question)}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Remove question ${index + 1}`}
            disabled={questions.length === 1}
            onClick={() => onChange(questions.filter((_, i) => i !== index))}
          >
            <Trash2 className="h-4 w-4 text-niq-red" />
          </Button>
        </div>
      ))}
      <Button
        variant="secondary"
        size="sm"
        disabled={questions.length >= maxQuestions}
        onClick={() => onChange([...questions, ""])}
      >
        <Plus className="h-4 w-4" /> Add question
      </Button>
    </div>
  );
}
