"""Pytest fixtures for InsightPulse test suite.

Provides shared fixtures for:
    - Sample panelist data
    - Survey questions
    - Mock LLM responses
    - Pipeline state objects
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def sample_panelist() -> dict[str, Any]:
    """A single sample panelist for unit tests."""
    return {
        "panelist_id": "HH_TEST_001",
        "age_group": "25-34",
        "income_group": "middle",
        "region": "south",
        "household_size": "3-4",
        "has_children": True,
        "education_level": "bachelors",
        "cluster_id": 0,
        "expansion_factor": 500.0,
    }


@pytest.fixture
def sample_panelists() -> list[dict[str, Any]]:
    """A small cohort of sample panelists covering all clusters."""
    return [
        {
            "panelist_id": f"HH_TEST_{i:03d}",
            "age_group": age,
            "income_group": income,
            "region": region,
            "household_size": "2",
            "has_children": False,
            "education_level": "bachelors",
            "cluster_id": i % 5,
            "expansion_factor": 500.0,
        }
        for i, (age, income, region) in enumerate([
            ("18-24", "low", "northeast"),
            ("25-34", "middle", "south"),
            ("35-44", "upper_middle", "west"),
            ("45-54", "high", "midwest"),
            ("55-64", "lower_middle", "south"),
        ])
    ]


@pytest.fixture
def sample_question() -> dict[str, Any]:
    """A sample survey question for testing."""
    return {
        "question_id": "q_1",
        "text": "How important is organic labeling when purchasing snacks?",
        "question_type": "likert_5",
        "options": [
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree",
        ],
        "category": "product_satisfaction",
        "context": "US snack food market",
        "is_sequential": False,
        "prior_questions": [],
    }


@pytest.fixture
def sample_response() -> dict[str, Any]:
    """A sample validated survey response."""
    return {
        "response_id": "HH_TEST_001_q_1",
        "question_id": "q_1",
        "panelist_id": "HH_TEST_001",
        "answer": "Agree",
        "reasoning": "As a young parent, I care about what goes into my family's food.",
        "confidence": 0.8,
        "model_used": "claude-sonnet-4-6",
        "generation_time_ms": 450.0,
        "token_count": 180,
        "cost_usd": 0.0003,
        "behavioral_cluster": 0,
        "demographic_summary": "25-34 | middle | south",
        "is_valid": True,
        "validation_flags": [],
    }


@pytest.fixture
def sample_pipeline_state(
    sample_question: dict[str, Any],
    sample_panelists: list[dict[str, Any]],
) -> dict[str, Any]:
    """A fully populated pipeline state for integration tests."""
    return {
        "raw_questions": [sample_question["text"]],
        "question_context": "US snack food market",
        "requested_cohort_size": 5,
        "requested_models": ["claude-sonnet-4-6"],
        "cohort_filters": {},
        "random_seed": 42,
        "status": "running",
        "parsed_questions": [sample_question],
        "selected_panelist_ids": [p["panelist_id"] for p in sample_panelists],
    }
