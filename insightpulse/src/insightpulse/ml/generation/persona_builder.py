"""L3 persona prompts — structured persona construction per twin."""


from __future__ import annotations

from typing import Any

from insightpulse.config.settings import GenerationConfig
from insightpulse.utils.logging import get_logger

logger = get_logger(__name__)

# Behavioral archetype descriptions used in persona prompts. Keys match the
# K=5 clusters from L2 (thesis Results-1) and the sample-data archetypes.
CLUSTER_PERSONAS: dict[int, str] = {
    0: (
        "You are a price-sensitive shopper. You carefully compare prices, "
        "frequently buy store brands, wait for sales, and use coupons regularly. "
        "You prioritize getting the best value for money and are willing to "
        "switch brands to save money."
    ),
    1: (
        "You are a premium brand loyalist. You prefer well-known premium brands "
        "and are willing to pay more for perceived quality. You rarely switch "
        "brands based on price and value consistency and brand reputation."
    ),
    2: (
        "You are a category explorer. You enjoy trying new products and brands, "
        "purchase across many different categories, and are open to innovation. "
        "You are influenced by product novelty and variety."
    ),
    3: (
        "You are a promotion-driven buyer. You plan purchases around promotions, "
        "stock up during deals, and are highly responsive to advertising and "
        "in-store promotions. You may switch brands for a good promotion."
    ),
    4: (
        "You are a convenience-oriented consumer. You value convenience and "
        "time-saving, prefer ready-to-use products, shop frequently at nearby "
        "stores, and make quick purchasing decisions without extensive comparison."
    ),
}





class PersonaPromptBuilder:
    """Builds versioned persona prompts from panelist context.

    Single responsibility: (panelist, question, priors) -> (system, user)
    prompt pair. The template version is stamped by config and recorded in
    provenance, so prompt changes are visible in run comparisons — prompt
    drift is a real confound in LLM systems.

    Example:
        >>> builder = PersonaPromptBuilder(config)
        >>> system, user = builder.build(row, question, prior_responses)
    """

    def __init__(self, config: GenerationConfig) -> None:
        """Store the template version and limits."""
        self._config = config

    @property
    def template_version(self) -> str:
        """Version tag stamped into every run's provenance."""
        return self._config.prompt_template_version

    def build(
        self,
        panelist: dict[str, Any],
        question: dict[str, Any],
        prior_responses: list[str] | None = None,
    ) -> tuple[str, str]:
        """Construct the (system, user) prompt pair for one twin response.

        Args:
            panelist: Panelist attributes (demographics + cluster).
            question: Structured question dict.
            prior_responses: The twin's earlier answers in this survey
                (sequential conditioning — evaluator feedback #3).

        Returns:
            (system_prompt, user_prompt).
        """
        cluster_id = int(panelist.get("cluster_id", 0) or 0)
        behavioral = CLUSTER_PERSONAS.get(cluster_id, CLUSTER_PERSONAS[0])

        demographics = (
            f"You are a {panelist.get('age_group', '25-34')} year old person "
            f"from the {panelist.get('region', 'suburban')} United States. "
            f"Your household has {panelist.get('household_size', '2')} members"
        )
        if panelist.get("has_children"):
            demographics += " including children under 18"
        demographics += (
            f". Your education level is {panelist.get('education_level', 'bachelors')} "
            f"and your household income is {panelist.get('income_group', 'middle')}."
        )

        system = f"""You are a synthetic survey respondent — a digital twin of a real consumer.
Answer the survey question AS THIS PERSON, based on the profile below.
[persona template {self.template_version}]

DEMOGRAPHIC PROFILE:
{demographics}

BEHAVIORAL PROFILE:
{behavioral}

RESPONSE RULES:
1. Answer ONLY from the provided options (if options are given).
2. Stay consistent with your demographic and behavioral profile.
3. Be realistic — real consumers don't always choose the most "correct" answer.
4. Give brief reasoning (1-2 sentences) for why this person answers this way.
5. Rate your confidence from 0.0 to 1.0.

Respond ONLY in this JSON format:
{{"answer": "...", "reasoning": "...", "confidence": 0.X}}"""

        user = f"Survey Question: {question['text']}"
        if question.get("options"):
            option_lines = "\n".join(
                f"  {i + 1}. {opt}" for i, opt in enumerate(question["options"])
            )
            user += f"\n\nResponse Options:\n{option_lines}"
        if question.get("context"):
            user += f"\n\nContext: {question['context']}"
        if prior_responses:
            prior_lines = "\n".join(
                f"  Q{i + 1}: {answer}" for i, answer in enumerate(prior_responses)
            )
            user += (
                f"\n\nYour previous answers in this survey:\n{prior_lines}"
                "\n\nEnsure your answer is logically consistent with your "
                "previous responses."
            )
        return system, user
